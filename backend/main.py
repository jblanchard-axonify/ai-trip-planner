from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import time
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Minimal observability via Arize/OpenInference (optional)
try:
    from arize.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.litellm import LiteLLMInstrumentor
    from openinference.instrumentation import using_prompt_template, using_metadata, using_attributes
    from opentelemetry import trace
    _TRACING = True
except Exception:
    def using_prompt_template(**kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    def using_metadata(*args, **kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    def using_attributes(*args, **kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    _TRACING = False

# LangGraph + LangChain
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
import httpx


class MealPlanRequest(BaseModel):
    dietary_restrictions: Optional[str] = None
    preferences: Optional[str] = None
    budget: Optional[str] = None
    num_people: int = 2
    days: int = 7
    health_goals: Optional[str] = None
    # Optional fields for enhanced session tracking and observability
    user_input: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    turn_index: Optional[int] = None


class MealPlanResponse(BaseModel):
    result: str
    tool_calls: List[Dict[str, Any]] = []


def _init_llm():
    # Simple, test-friendly LLM init
    class _Fake:
        def __init__(self):
            pass
        def bind_tools(self, tools):
            return self
        def invoke(self, messages):
            class _Msg:
                content = "Test itinerary"
                tool_calls: List[Dict[str, Any]] = []
            return _Msg()

    if os.getenv("TEST_MODE"):
        return _Fake()
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=1500)
    elif os.getenv("OPENROUTER_API_KEY"):
        # Use OpenRouter via OpenAI-compatible client
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            temperature=0.7,
        )
    else:
        # Require a key unless running tests
        raise ValueError("Please set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env")


llm = _init_llm()


# Feature flag for optional RAG demo (opt-in for learning)
ENABLE_RAG = os.getenv("ENABLE_RAG", "0").lower() not in {"0", "false", "no"}


# RAG helper: Load curated recipes as LangChain documents
def _load_recipe_documents(path: Path) -> List[Document]:
    """Load recipes JSON and convert to LangChain Documents."""
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return []

    docs: List[Document] = []
    for recipe in raw:
        name = recipe.get("name")
        if not name:
            continue
        cuisine = recipe.get("cuisine", "")
        dietary_tags = recipe.get("dietary_tags", []) or []
        meal_type = recipe.get("meal_type", []) or []
        nutrition = recipe.get("nutrition", {})
        
        metadata = {
            "name": name,
            "cuisine": cuisine,
            "dietary_tags": dietary_tags,
            "meal_type": meal_type,
            "prep_time": recipe.get("prep_time", 0),
            "cook_time": recipe.get("cook_time", 0),
            "difficulty": recipe.get("difficulty", "medium"),
            "calories": nutrition.get("calories", 0),
            "protein": nutrition.get("protein", 0),
        }
        
        # Build searchable content with dietary tags and meal types
        tags_text = ", ".join(dietary_tags) if dietary_tags else "no restrictions"
        meals_text = ", ".join(meal_type) if meal_type else "any meal"
        ingredients_list = [ing.get("item", "") for ing in recipe.get("ingredients", [])]
        ingredients_text = ", ".join(ingredients_list[:8])  # First 8 ingredients
        
        content = (
            f"Recipe: {name}\n"
            f"Cuisine: {cuisine}\n"
            f"Dietary tags: {tags_text}\n"
            f"Meal type: {meals_text}\n"
            f"Ingredients: {ingredients_text}\n"
            f"Calories: {nutrition.get('calories', 0)}, Protein: {nutrition.get('protein', 0)}g"
        )
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


class RecipeRetriever:
    """Retrieves curated recipes using vector similarity search.
    
    This class demonstrates production RAG patterns for students:
    - Vector embeddings for semantic search
    - Fallback to keyword matching when embeddings unavailable
    - Graceful degradation with feature flags
    """
    
    def __init__(self, data_path: Path):
        """Initialize retriever with recipe data.
        
        Args:
            data_path: Path to recipes.json file
        """
        self._docs = _load_recipe_documents(data_path)
        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._vectorstore: Optional[InMemoryVectorStore] = None
        
        # Only create embeddings when RAG is enabled and we have an API key
        if ENABLE_RAG and self._docs and not os.getenv("TEST_MODE"):
            try:
                model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
                self._embeddings = OpenAIEmbeddings(model=model)
                store = InMemoryVectorStore(embedding=self._embeddings)
                store.add_documents(self._docs)
                self._vectorstore = store
            except Exception:
                # Gracefully degrade to keyword search if embeddings fail
                self._embeddings = None
                self._vectorstore = None

    @property
    def is_empty(self) -> bool:
        """Check if any documents were loaded."""
        return not self._docs

    def retrieve(self, preferences: Optional[str], dietary_restrictions: Optional[str], *, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant recipes based on preferences and restrictions.
        
        Args:
            preferences: Cuisine preferences (e.g., "Italian, Asian")
            dietary_restrictions: Dietary restrictions (e.g., "vegetarian, gluten-free")
            k: Number of results to return
            
        Returns:
            List of dicts with 'content', 'metadata', and 'score' keys
        """
        if not ENABLE_RAG or self.is_empty:
            return []

        # Use vector search if available, otherwise fall back to keywords
        if not self._vectorstore:
            return self._keyword_fallback(preferences, dietary_restrictions, k=k)

        query_parts = []
        if preferences:
            query_parts.append(f"cuisine: {preferences}")
        if dietary_restrictions:
            query_parts.append(f"dietary: {dietary_restrictions}")
        query = " ".join(query_parts) if query_parts else "healthy meal"
        
        try:
            # LangChain retriever ensures embeddings + searches are traced
            retriever = self._vectorstore.as_retriever(search_kwargs={"k": max(k, 8)})
            docs = retriever.invoke(query)
        except Exception:
            return self._keyword_fallback(preferences, dietary_restrictions, k=k)

        # Format results with metadata and scores
        top_docs = docs[:k]
        results = []
        for doc in top_docs:
            score_val: float = 0.0
            if isinstance(doc.metadata, dict):
                maybe_score = doc.metadata.get("score")
                if isinstance(maybe_score, (int, float)):
                    score_val = float(maybe_score)
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score_val,
            })

        if not results:
            return self._keyword_fallback(preferences, dietary_restrictions, k=k)
        return results

    def _keyword_fallback(self, preferences: Optional[str], dietary_restrictions: Optional[str], *, k: int) -> List[Dict[str, Any]]:
        """Simple keyword-based retrieval when embeddings unavailable.
        
        This demonstrates graceful degradation for students learning about
        fallback strategies in production systems.
        """
        pref_terms = [part.strip().lower() for part in (preferences or "").split(",") if part.strip()]
        diet_terms = [part.strip().lower() for part in (dietary_restrictions or "").split(",") if part.strip()]

        def _score(doc: Document) -> int:
            score = 0
            content_lower = doc.page_content.lower()
            dietary_tags = [tag.lower() for tag in doc.metadata.get("dietary_tags", [])]
            cuisine_lower = doc.metadata.get("cuisine", "").lower()
            
            # Match dietary restrictions (high priority)
            for term in diet_terms:
                if term and term in " ".join(dietary_tags):
                    score += 3
                if term and term in content_lower:
                    score += 2
            
            # Match cuisine preferences
            for term in pref_terms:
                if term and term in cuisine_lower:
                    score += 2
                if term and term in content_lower:
                    score += 1
            
            return score

        scored_docs = [(_score(doc), doc) for doc in self._docs]
        scored_docs.sort(key=lambda item: item[0], reverse=True)
        top_docs = scored_docs[:k]
        
        results = []
        for score, doc in top_docs:
            if score > 0:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                })
        return results


# Initialize retriever at module level (loads data once at startup)
_DATA_DIR = Path(__file__).parent / "data"
RECIPE_RETRIEVER = RecipeRetriever(_DATA_DIR / "recipes.json")


# Search API configuration and helpers
SEARCH_TIMEOUT = 10.0  # seconds


def _compact(text: str, limit: int = 200) -> str:
    """Compact text to a maximum length, truncating at word boundaries."""
    if not text:
        return ""
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    truncated = cleaned[:limit]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated.rstrip(",.;- ")


def _search_api(query: str) -> Optional[str]:
    """Search the web using Tavily or SerpAPI if configured, return None otherwise.
    
    This demonstrates graceful degradation: tools work with or without API keys.
    Students can enable real search by adding TAVILY_API_KEY or SERPAPI_API_KEY.
    """
    query = query.strip()
    if not query:
        return None

    # Try Tavily first (recommended for AI apps)
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            with httpx.Client(timeout=SEARCH_TIMEOUT) as client:
                resp = client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_key,
                        "query": query,
                        "max_results": 3,
                        "search_depth": "basic",
                        "include_answer": True,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("answer") or ""
                snippets = [
                    item.get("content") or item.get("snippet") or ""
                    for item in data.get("results", [])
                ]
                combined = " ".join([answer] + snippets).strip()
                if combined:
                    return _compact(combined)
        except Exception:
            pass  # Fail gracefully, try next option

    # Try SerpAPI as fallback
    serp_key = os.getenv("SERPAPI_API_KEY")
    if serp_key:
        try:
            with httpx.Client(timeout=SEARCH_TIMEOUT) as client:
                resp = client.get(
                    "https://serpapi.com/search",
                    params={
                        "api_key": serp_key,
                        "engine": "google",
                        "num": 5,
                        "q": query,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                organic = data.get("organic_results", [])
                snippets = [item.get("snippet", "") for item in organic]
                combined = " ".join(snippets).strip()
                if combined:
                    return _compact(combined)
        except Exception:
            pass  # Fail gracefully

    return None  # No search APIs configured


def _llm_fallback(instruction: str, context: Optional[str] = None) -> str:
    """Use the LLM to generate a response when search APIs aren't available.
    
    This ensures tools always return useful information, even without API keys.
    """
    prompt = "Respond with 200 characters or less.\n" + instruction.strip()
    if context:
        prompt += "\nContext:\n" + context.strip()
    response = llm.invoke([
        SystemMessage(content="You are a concise meal planning assistant."),
        HumanMessage(content=prompt),
    ])
    return _compact(response.content)


def _with_prefix(prefix: str, summary: str) -> str:
    """Add a prefix to a summary for clarity."""
    text = f"{prefix}: {summary}" if prefix else summary
    return _compact(text)


# Tools with real API calls + LLM fallback (graceful degradation pattern)
@tool
def calculate_nutrition(dietary_restrictions: Optional[str] = None, health_goals: Optional[str] = None, num_people: int = 2) -> str:
    """Calculate daily nutritional targets based on health goals and restrictions."""
    query = f"daily nutrition targets {health_goals or 'balanced diet'} {dietary_restrictions or ''} for {num_people} people"
    summary = _search_api(query)
    if summary:
        return _with_prefix("Nutrition targets", summary)
    
    instruction = f"Provide daily calorie and macro targets (protein, carbs, fat) for {num_people} people with goals: {health_goals or 'maintenance'} and restrictions: {dietary_restrictions or 'none'}."
    return _llm_fallback(instruction)


@tool
def get_dietary_guidelines(restrictions: str) -> str:
    """Get dietary guidelines and considerations for specific restrictions or allergies."""
    query = f"dietary guidelines {restrictions} meal planning nutrition advice"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{restrictions} guidelines", summary)
    
    instruction = f"Explain key nutritional considerations and guidelines for someone with {restrictions}."
    return _llm_fallback(instruction)


@tool
def search_recipes(cuisine: Optional[str] = None, dietary_tags: Optional[str] = None) -> str:
    """Search for recipes matching cuisine preferences and dietary requirements."""
    query = f"{cuisine or 'varied'} recipes {dietary_tags or 'healthy'} meal ideas"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{cuisine or 'Recipe'} ideas", summary)
    
    instruction = f"Suggest recipe ideas for {cuisine or 'various cuisines'} that are {dietary_tags or 'healthy and balanced'}."
    return _llm_fallback(instruction)


@tool
def get_ingredient_substitutions(ingredient: str, reason: str = "preference") -> str:
    """Find ingredient substitutions for allergies, preferences, or availability."""
    query = f"substitute for {ingredient} {reason} cooking alternatives"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"Substitute {ingredient}", summary)
    
    instruction = f"Suggest substitutes for {ingredient} due to {reason}."
    return _llm_fallback(instruction)


@tool
def estimate_grocery_costs(items: str, num_people: int = 2, days: int = 7) -> str:
    """Estimate grocery costs for meal plan."""
    query = f"average grocery cost {items} for {num_people} people {days} days"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{days}-day grocery budget", summary)
    
    instruction = f"Estimate grocery costs for {num_people} people for {days} days including {items}."
    return _llm_fallback(instruction)


@tool
def find_budget_alternatives(ingredient: str) -> str:
    """Find budget-friendly alternatives to expensive ingredients."""
    query = f"cheap alternative to {ingredient} budget grocery substitutes"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"Budget alternative to {ingredient}", summary)
    
    instruction = f"Suggest affordable alternatives to {ingredient} that work well in cooking."
    return _llm_fallback(instruction)


@tool
def organize_shopping_list(items: str) -> str:
    """Organize shopping list by grocery store sections."""
    instruction = f"Organize these items by grocery store section (produce, dairy, meat, pantry, frozen): {items}"
    return _llm_fallback(instruction)


@tool
def identify_pantry_staples(recipes: str) -> str:
    """Identify which ingredients are pantry staples vs. fresh items."""
    instruction = f"From these recipe ingredients, identify pantry staples vs fresh items: {recipes}"
    return _llm_fallback(instruction)


class MealPlanState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    meal_request: Dict[str, Any]
    nutrition: Optional[str]
    recipes: Optional[str]
    budget: Optional[str]
    grocery: Optional[str]
    final_plan: Optional[str]
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]


def nutrition_agent(state: MealPlanState) -> MealPlanState:
    req = state["meal_request"]
    dietary_restrictions = req.get("dietary_restrictions", "none")
    health_goals = req.get("health_goals", "balanced diet")
    num_people = req.get("num_people", 2)
    
    prompt_t = (
        "You are a nutrition expert.\n"
        "Analyze dietary needs for {num_people} people with restrictions: {dietary_restrictions}.\n"
        "Health goals: {health_goals}.\n"
        "Use tools to calculate nutrition targets and get dietary guidelines, then summarize."
    )
    vars_ = {
        "num_people": num_people,
        "dietary_restrictions": dietary_restrictions,
        "health_goals": health_goals
    }
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [calculate_nutrition, get_dietary_guidelines]
    agent = llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    
    # Agent metadata and prompt template instrumentation
    with using_attributes(tags=["nutrition", "health_analysis"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "nutrition")
                current_span.set_attribute("metadata.agent_node", "nutrition_agent")
        
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)
    
    # Collect tool calls and execute them
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "nutrition", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        
        # Add tool results and ask for synthesis
        messages.append(res)
        messages.extend(tr["messages"])
        
        synthesis_prompt = f"Provide a comprehensive nutritional summary for {num_people} people with {dietary_restrictions} restrictions and {health_goals} goals."
        messages.append(SystemMessage(content=synthesis_prompt))
        
        # Instrument synthesis LLM call
        synthesis_vars = {"num_people": num_people, "dietary_restrictions": dietary_restrictions, "health_goals": health_goals}
        with using_prompt_template(template=synthesis_prompt, variables=synthesis_vars, version="v1-synthesis"):
            final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    return {"messages": [SystemMessage(content=out)], "nutrition": out, "tool_calls": calls}


def budget_agent(state: MealPlanState) -> MealPlanState:
    req = state["meal_request"]
    num_people = req.get("num_people", 2)
    days = req.get("days", 7)
    budget = req.get("budget", "moderate")
    
    prompt_t = (
        "You are a budget analyst.\n"
        "Estimate grocery costs for {num_people} people over {days} days with a {budget} budget.\n"
        "Use tools to estimate costs and suggest budget-friendly alternatives."
    )
    vars_ = {"num_people": num_people, "days": days, "budget": budget}
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [estimate_grocery_costs, find_budget_alternatives]
    agent = llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    
    # Agent metadata and prompt template instrumentation
    with using_attributes(tags=["budget", "cost_analysis"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "budget")
                current_span.set_attribute("metadata.agent_node", "budget_agent")
        
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)
    
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "budget", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        
        # Add tool results and ask for synthesis
        messages.append(res)
        messages.extend(tr["messages"])
        
        synthesis_prompt = f"Create a detailed grocery budget estimate for {num_people} people over {days} days with a {budget} budget."
        messages.append(SystemMessage(content=synthesis_prompt))
        
        # Instrument synthesis LLM call
        synthesis_vars = {"num_people": num_people, "days": days, "budget": budget}
        with using_prompt_template(template=synthesis_prompt, variables=synthesis_vars, version="v1-synthesis"):
            final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    return {"messages": [SystemMessage(content=out)], "budget": out, "tool_calls": calls}


def recipe_agent(state: MealPlanState) -> MealPlanState:
    req = state["meal_request"]
    preferences = req.get("preferences", "varied cuisines")
    dietary_restrictions = req.get("dietary_restrictions", "none")
    days = req.get("days", 7)
    
    # RAG: Retrieve curated recipes if enabled
    context_lines = []
    if ENABLE_RAG:
        retrieved = RECIPE_RETRIEVER.retrieve(preferences, dietary_restrictions, k=5)
        if retrieved:
            context_lines.append("=== Curated Recipes (from database) ===")
            for idx, item in enumerate(retrieved, 1):
                content = item["content"]
                name = item["metadata"].get("name", "Recipe")
                context_lines.append(f"{idx}. {content}")
            context_lines.append("=== End of Curated Recipes ===\n")
    
    context_text = "\n".join(context_lines) if context_lines else ""
    
    prompt_t = (
        "You are a chef and recipe specialist.\n"
        "Find recipes for a {days}-day meal plan matching preferences: {preferences}.\n"
        "Dietary restrictions: {dietary_restrictions}. Use tools to search for recipes.\n"
        "Ensure variety across cuisines and cooking methods.\n"
    )
    
    # Add retrieved context to prompt if available
    if context_text:
        prompt_t += "\nRelevant curated recipes from our database:\n{context}\n"
    
    vars_ = {
        "preferences": preferences,
        "dietary_restrictions": dietary_restrictions,
        "days": days,
        "context": context_text if context_text else "No curated recipes available.",
    }
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [search_recipes, get_ingredient_substitutions]
    agent = llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    
    # Agent metadata and prompt template instrumentation
    with using_attributes(tags=["recipe", "meal_selection"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "recipe")
                current_span.set_attribute("metadata.agent_node", "recipe_agent")
                if ENABLE_RAG and context_text:
                    current_span.set_attribute("metadata.rag_enabled", "true")
        
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)
    
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "recipe", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        
        # Add tool results and ask for synthesis
        messages.append(res)
        messages.extend(tr["messages"])
        
        synthesis_prompt = f"Create a curated list of {days * 3} recipes (breakfast, lunch, dinner) matching {preferences} and {dietary_restrictions}."
        messages.append(SystemMessage(content=synthesis_prompt))
        
        # Instrument synthesis LLM call
        synthesis_vars = {"preferences": preferences, "dietary_restrictions": dietary_restrictions, "days": days}
        with using_prompt_template(template=synthesis_prompt, variables=synthesis_vars, version="v1-synthesis"):
            final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    return {"messages": [SystemMessage(content=out)], "recipes": out, "tool_calls": calls}


def grocery_agent(state: MealPlanState) -> MealPlanState:
    req = state["meal_request"]
    num_people = req.get("num_people", 2)
    days = req.get("days", 7)
    
    # Get recipes from previous agent
    recipes_info = state.get("recipes", "")
    
    prompt_t = (
        "You are a grocery shopping expert.\n"
        "Create a consolidated shopping list for {num_people} people over {days} days.\n"
        "Organize items by store section and identify pantry staples vs fresh items.\n"
        "Use the recipes provided to extract ingredients.\n"
    )
    vars_ = {"num_people": num_people, "days": days}
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    if recipes_info:
        messages.append(SystemMessage(content=f"Recipes to shop for:\n{recipes_info[:500]}"))
    
    tools = [organize_shopping_list, identify_pantry_staples]
    agent = llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    
    # Agent metadata and prompt template instrumentation
    with using_attributes(tags=["grocery", "shopping_list"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "grocery")
                current_span.set_attribute("metadata.agent_node", "grocery_agent")
        
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)
    
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "grocery", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        
        # Add tool results and ask for synthesis
        messages.append(res)
        messages.extend(tr["messages"])
        
        synthesis_prompt = f"Create a well-organized shopping list for {num_people} people for {days} days, organized by store section."
        messages.append(SystemMessage(content=synthesis_prompt))
        
        # Instrument synthesis LLM call
        synthesis_vars = {"num_people": num_people, "days": days}
        with using_prompt_template(template=synthesis_prompt, variables=synthesis_vars, version="v1-synthesis"):
            final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    return {"messages": [SystemMessage(content=out)], "grocery": out, "tool_calls": calls}


def meal_plan_synthesizer(state: MealPlanState) -> MealPlanState:
    req = state["meal_request"]
    num_people = req.get("num_people", 2)
    days = req.get("days", 7)
    dietary_restrictions = req.get("dietary_restrictions", "none")
    preferences = req.get("preferences", "varied")
    user_input = (req.get("user_input") or "").strip()
    
    prompt_parts = [
        "Create a comprehensive {days}-day meal plan for {num_people} people.",
        "Dietary restrictions: {dietary_restrictions}",
        "Preferences: {preferences}",
        "",
        "Inputs from specialist agents:",
        "Nutrition: {nutrition}",
        "Recipes: {recipes}",
        "Budget: {budget}",
        "Shopping List: {grocery}",
        "",
        "Synthesize all inputs into a day-by-day meal plan with:",
        "- Breakfast, lunch, dinner for each day",
        "- Nutritional balance across the week",
        "- Shopping list organized by category",
        "- Weekly prep tips and timing guidance",
    ]
    if user_input:
        prompt_parts.append("Additional user notes: {user_input}")
    
    prompt_t = "\n".join(prompt_parts)
    vars_ = {
        "days": days,
        "num_people": num_people,
        "dietary_restrictions": dietary_restrictions,
        "preferences": preferences,
        "nutrition": (state.get("nutrition") or "")[:400],
        "recipes": (state.get("recipes") or "")[:400],
        "budget": (state.get("budget") or "")[:400],
        "grocery": (state.get("grocery") or "")[:400],
        "user_input": user_input,
    }
    
    # Add span attributes for better observability in Arize
    # NOTE: using_attributes must be OUTER context for proper propagation
    with using_attributes(tags=["meal_plan", "final_agent"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.meal_plan_synthesizer", "true")
                current_span.set_attribute("metadata.agent_type", "meal_plan")
                current_span.set_attribute("metadata.agent_node", "meal_plan_synthesizer")
                if user_input:
                    current_span.set_attribute("metadata.user_input", user_input)
        
        # Prompt template wrapper for Arize Playground integration
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = llm.invoke([SystemMessage(content=prompt_t.format(**vars_))])
    
    return {"messages": [SystemMessage(content=res.content)], "final_plan": res.content}


def build_graph():
    g = StateGraph(MealPlanState)
    g.add_node("nutrition_node", nutrition_agent)
    g.add_node("recipe_node", recipe_agent)
    g.add_node("budget_node", budget_agent)
    g.add_node("grocery_node", grocery_agent)
    g.add_node("synthesizer_node", meal_plan_synthesizer)

    # Run nutrition, recipe, budget, and grocery agents in parallel
    g.add_edge(START, "nutrition_node")
    g.add_edge(START, "recipe_node")
    g.add_edge(START, "budget_node")
    g.add_edge(START, "grocery_node")
    
    # All four agents feed into the meal plan synthesizer
    g.add_edge("nutrition_node", "synthesizer_node")
    g.add_edge("recipe_node", "synthesizer_node")
    g.add_edge("budget_node", "synthesizer_node")
    g.add_edge("grocery_node", "synthesizer_node")
    
    g.add_edge("synthesizer_node", END)

    # Compile without checkpointer to avoid state persistence issues
    return g.compile()


app = FastAPI(title="SmartMeal Planner")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_frontend():
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "frontend", "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return {"message": "frontend/index.html not found"}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "smartmeal-planner"}


# Initialize tracing once at startup, not per request
if _TRACING:
    try:
        space_id = os.getenv("ARIZE_SPACE_ID")
        api_key = os.getenv("ARIZE_API_KEY")
        if space_id and api_key:
            tp = register(space_id=space_id, api_key=api_key, project_name="smartmeal-planner")
            LangChainInstrumentor().instrument(tracer_provider=tp, include_chains=True, include_agents=True, include_tools=True)
            LiteLLMInstrumentor().instrument(tracer_provider=tp, skip_dep_check=True)
    except Exception:
        pass

@app.post("/plan-meal", response_model=MealPlanResponse)
def plan_meal(req: MealPlanRequest):
    graph = build_graph()
    
    # Only include necessary fields in initial state
    # Agent outputs (nutrition, recipes, budget, grocery, final_plan) will be added during execution
    state = {
        "messages": [],
        "meal_request": req.model_dump(),
        "tool_calls": [],
    }
    
    # Add session and user tracking attributes to the trace
    session_id = req.session_id
    user_id = req.user_id
    turn_idx = req.turn_index
    
    # Build attributes for session and user tracking
    attrs_kwargs = {}
    if session_id:
        attrs_kwargs["session_id"] = session_id
    if user_id:
        attrs_kwargs["user_id"] = user_id
    
    # Add turn_index as a custom span attribute if provided
    if turn_idx is not None and _TRACING:
        with using_attributes(**attrs_kwargs):
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("turn_index", turn_idx)
            out = graph.invoke(state)
    else:
        with using_attributes(**attrs_kwargs):
            out = graph.invoke(state)
    
    return MealPlanResponse(result=out.get("final_plan", ""), tool_calls=out.get("tool_calls", []))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
