"""
LangGraph Question-Answering Agent with SEPARATE MCP Nodes & Tool Calls for:
  1. `postgres_mcp_node` -> PostgreSQL Database (`postgres_list_tables`, `postgres_describe_table`, `postgres_run_query`)
  2. `website_mcp_node`  -> Live Company Website HTML (`website_list_pages`, `website_search_pages`, `website_fetch_page`)
  3. `pdf_mcp_node`      -> Website Policy PDFs (`pdf_list_documents`, `pdf_search_documents`, `pdf_read_document`)
  4. `minio_mcp_node`    -> MinIO Object Storage & CSVs (`minio_list_objects`, `minio_query_csv`, `minio_read_object`)
  5. `elastic_mcp_node`  -> Elasticsearch search over the MinIO documents (`elastic_index_stats`, `elastic_search_documents`)
                            then reads the matched object back from MinIO (`minio_read_object`)

Every execution logs an MLflow Run and a nested MLflow Trace (`http://localhost:5000`).
"""

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Literal, TypedDict

from dotenv import load_dotenv

from mcp_server import (
    HAS_MLFLOW,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    configure_mlflow,
    elastic_index_stats,
    elastic_search_documents,
    minio_list_objects,
    minio_query_csv,
    minio_read_object,
    minio_search_markdown_sources,
    mlflow,
    mlflow_trace,
    pdf_list_documents,
    pdf_read_document,
    pdf_search_documents,
    postgres_describe_table,
    postgres_list_tables,
    postgres_run_query,
    website_fetch_page,
    website_list_pages,
    website_search_pages,
)

try:
    from langgraph.graph import END, START, StateGraph
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False

load_dotenv()
configure_mlflow()

RouteType = Literal["postgres", "website", "pdf", "minio", "elastic"]


def is_elastic_question(q: str) -> bool:
    q = q.lower()
    return any(k in q for k in ["elastic", "full-text", "full text", "search index", "keyword search"])


_POLICY_HINTS = [
    "policy", "policies", "procedure", "regulation", "terms and conditions", "disclosure", "handbook",
    "code of conduct", "claims", "tariff", "surcharge", "demurrage", "detention", "reefer", "customs",
    "cargo", "dangerous goods", "hazardous", "reg e", "reg cc", "hipaa", "bsa", "aml", "ofac", "kyc",
    "privacy", "charity", "financial assistance", "dispute", "refund", "overdraft", "fee schedule",
    "fund transfer", "liability", "infection", "compliance", "patient", "telehealth", "eligibility",
    "what does", "what is the rule", "according to",
]
_DB_ENTITY_HINTS = [
    "order", "customer", "product", "categor", "payment", "inventory", "warehouse", "batch", "etl",
    "revenue", "spending", "stock",
]
_CSV_HINTS = [
    "csv", "indian_tech_jobs", "tech job", "job posting", "data scientist", "bengaluru", "bangalore", "salary",
]


def route_question(question: str) -> RouteType:
    """Pick the MCP source for a question.

    1. A source named explicitly in the question is honoured (Elasticsearch / PDF / website / PostgreSQL / CSV).
    2. Otherwise policy / document questions go to Elasticsearch (the search index over MinIO).
    3. Otherwise table-style questions go to PostgreSQL.
    4. Anything unrecognised is treated as a document question -> Elasticsearch.
    """
    q = question.lower()

    # 1. explicitly named source
    if is_elastic_question(q):
        return "elastic"
    if "pdf" in q:
        return "pdf"
    if any(k in q for k in ["website", "web page", "webpage", "faq page", "about us page", "localhost:8080"]):
        return "website"
    if any(k in q for k in ["postgres", "sql", "database", " db ", "table"]):
        return "postgres"
    if any(k in q for k in _CSV_HINTS):
        return "minio"

    # 2. policy / document questions -> Elasticsearch over MinIO
    if any(k in q for k in _POLICY_HINTS):
        return "elastic"

    # 3. data questions -> PostgreSQL
    if any(k in q for k in _DB_ENTITY_HINTS):
        return "postgres"

    # 4. default
    return "elastic"


# =====================================================================
# 1. DEFINE LANGGRAPH STATE
# =====================================================================

class QAState(TypedDict):
    question: str
    route: RouteType
    mcp_tool_calls: List[str]
    source_context: str
    final_answer: str


# =====================================================================
# 2. HELPER: OPTIONAL LLM CALL + SQL BUILDER
# =====================================================================

def _call_llm_if_available(system_prompt: str, user_prompt: str) -> str | None:
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass

    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            resp = client.messages.create(
                model=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return resp.content[0].text.strip()
        except Exception:
            pass

    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            resp = client.models.generate_content(
                model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
                contents=f"{system_prompt}\n\n{user_prompt}",
            )
            return resp.text.strip()
        except Exception:
            pass

    return None


def _build_sql_for_question(question: str, schema_info: str) -> str:
    llm_sql = _call_llm_if_available(
        system_prompt=(
            "You are a PostgreSQL expert. Given the schema below, return ONLY a valid read-only "
            "PostgreSQL SELECT query (no markdown fences, no commentary).\n\nSchema:\n" + schema_info
        ),
        user_prompt=question,
    )
    if llm_sql:
        cleaned = re.sub(r"^```(?:sql)?\s*|\s*```$", "", llm_sql.strip(), flags=re.IGNORECASE)
        if cleaned.upper().startswith(("SELECT", "WITH")):
            return cleaned

    q = question.lower()
    if "batch" in q or "etl" in q:
        return "SELECT batch_name, batch_type, row_counts, loaded_at FROM etl_batch_control ORDER BY batch_name;"
    if "deliver" in q and "order" in q:
        return (
            "SELECT status, COUNT(*) AS order_count, ROUND(SUM(total_amount), 2) AS total_revenue "
            "FROM orders WHERE status = 'delivered' GROUP BY status;"
        )
    if "status" in q and "order" in q:
        return (
            "SELECT status, COUNT(*) AS order_count, ROUND(SUM(total_amount), 2) AS total_amount "
            "FROM orders GROUP BY status ORDER BY order_count DESC;"
        )
    if "payment" in q:
        return (
            "SELECT payment_method, status, COUNT(*) AS count, ROUND(SUM(amount), 2) AS total_amount "
            "FROM payments GROUP BY payment_method, status ORDER BY total_amount DESC LIMIT 10;"
        )
    if "categor" in q:
        return (
            "SELECT c.category_name, COUNT(p.product_id) AS product_count, ROUND(AVG(p.price), 2) AS avg_price "
            "FROM categories c LEFT JOIN products p ON c.category_id = p.category_id AND NOT p.is_deleted "
            "GROUP BY c.category_name ORDER BY product_count DESC;"
        )
    if "inventory" in q or "warehouse" in q or "stock" in q:
        return (
            "SELECT warehouse_location, COUNT(*) AS stocked_products, SUM(quantity_on_hand) AS total_units "
            "FROM inventory GROUP BY warehouse_location ORDER BY total_units DESC;"
        )
    if "customer" in q:
        return (
            "SELECT c.customer_id, c.first_name || ' ' || c.last_name AS customer_name, c.city, "
            "COUNT(o.order_id) AS total_orders, ROUND(COALESCE(SUM(o.total_amount), 0), 2) AS total_spent "
            "FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id "
            "WHERE NOT c.is_deleted GROUP BY c.customer_id, c.first_name, c.last_name, c.city "
            "ORDER BY total_spent DESC LIMIT 5;"
        )
    if "product" in q:
        return (
            "SELECT p.product_id, p.product_name, p.price, SUM(oi.quantity) AS total_units_sold "
            "FROM products p JOIN order_items oi ON p.product_id = oi.product_id "
            "WHERE NOT p.is_deleted GROUP BY p.product_id, p.product_name, p.price "
            "ORDER BY total_units_sold DESC LIMIT 5;"
        )
    return (
        "SELECT "
        "(SELECT COUNT(*) FROM categories WHERE NOT is_deleted) AS active_categories, "
        "(SELECT COUNT(*) FROM products WHERE NOT is_deleted) AS active_products, "
        "(SELECT COUNT(*) FROM customers WHERE NOT is_deleted) AS active_customers, "
        "(SELECT COUNT(*) FROM orders) AS total_orders, "
        "(SELECT ROUND(SUM(total_amount), 2) FROM orders) AS gross_order_value;"
    )


# =====================================================================
# 3. FOUR SEPARATE LANGGRAPH MCP NODES
# =====================================================================

@mlflow_trace(name="langgraph.router_node", span_type="CHAIN")
def router_node(state: QAState) -> Dict[str, Any]:
    """Routes the question to a dedicated MCP Node: postgres, website, pdf, minio (CSV) or elastic (default for documents)."""
    route: RouteType = route_question(state["question"])

    return {"route": route, "mcp_tool_calls": [f"router_node -> routed to '{route}_mcp_node'"]}


@mlflow_trace(name="langgraph.postgres_mcp_node", span_type="AGENT")
def postgres_mcp_node(state: QAState) -> Dict[str, Any]:
    """[1. PostgreSQL MCP Node] Calls `postgres_list_tables`, `postgres_describe_table`, `postgres_run_query`."""
    logs = list(state.get("mcp_tool_calls", []))

    tables_json = postgres_list_tables()
    logs.append("MCP Call [PostgreSQL]: postgres_list_tables()")

    schema_json = postgres_describe_table("orders")
    logs.append("MCP Call [PostgreSQL]: postgres_describe_table('orders')")

    sql = _build_sql_for_question(state["question"], f"{tables_json}\n{schema_json}")
    res_json = postgres_run_query(sql)
    logs.append(f"MCP Call [PostgreSQL]: postgres_run_query({sql!r})")

    res = json.loads(res_json)
    summary = (
        f"[1. PostgreSQL MCP Calls -> `tinitiate-ai-postgres` (localhost:5432)]\n"
        f"• Executed SQL: `{sql}`\n"
        f"• Rows Returned ({res.get('row_count', 0)}):\n{json.dumps(res.get('rows', []), indent=2)}"
    )
    return {"source_context": summary, "mcp_tool_calls": logs}


@mlflow_trace(name="langgraph.website_mcp_node", span_type="AGENT")
def website_mcp_node(state: QAState) -> Dict[str, Any]:
    """[2. Website HTTP MCP Node] Calls `website_list_pages`, `website_search_pages`, `website_fetch_page`."""
    logs = list(state.get("mcp_tool_calls", []))
    question = state["question"]

    _ = website_list_pages()
    logs.append("MCP Call [Website HTTP]: website_list_pages()")

    search_json = website_search_pages(question)
    logs.append(f"MCP Call [Website HTTP]: website_search_pages({question!r})")
    search_data = json.loads(search_json)

    matches = search_data.get("top_matching_pages", [])
    if matches:
        top = matches[0]
        page_json = website_fetch_page(top["company"], top["page"])
        logs.append(f"MCP Call [Website HTTP]: website_fetch_page({top['company']!r}, {top['page']!r})")
        page_data = json.loads(page_json)
        summary = (
            f"[2. Website HTTP MCP Calls -> `tinitiate-ai-websites` ({top['url']})]\n"
            f"• Page Title: {page_data.get('title')}\n"
            f"• Headings on Page: {', '.join(page_data.get('headings', [])[:6])}\n"
            f"• Extracted Web Page Content:\n{top['matching_excerpt']}"
        )
    else:
        page_json = website_fetch_page("portal", "index.html")
        logs.append("MCP Call [Website HTTP]: website_fetch_page('portal', 'index.html')")
        page_data = json.loads(page_json)
        summary = (
            f"[2. Website HTTP MCP Calls -> `tinitiate-ai-websites`]\n"
            f"• Portal URL: {page_data.get('url')}\n"
            f"• Content:\n{page_data.get('page_text', '')[:800]}"
        )

    return {"source_context": summary, "mcp_tool_calls": logs}


@mlflow_trace(name="langgraph.pdf_mcp_node", span_type="AGENT")
def pdf_mcp_node(state: QAState) -> Dict[str, Any]:
    """[3. Website Policy PDF MCP Node] Calls `pdf_list_documents`, `pdf_search_documents`, `pdf_read_document`."""
    logs = list(state.get("mcp_tool_calls", []))
    question = state["question"]

    _ = pdf_list_documents()
    logs.append("MCP Call [Website PDF]: pdf_list_documents()")

    search_json = pdf_search_documents(question)
    logs.append(f"MCP Call [Website PDF]: pdf_search_documents({question!r})")
    search_data = json.loads(search_json)

    pdf_matches = search_data.get("top_pdf_matches", [])
    if pdf_matches:
        top = pdf_matches[0]
        _ = pdf_read_document(top["website_pdf_url"], max_chars=3000)
        logs.append(f"MCP Call [Website PDF]: pdf_read_document({top['website_pdf_url']!r})")
        summary = (
            f"[3. Website Policy PDF MCP Calls -> `{top['pdf_name']}` (parsed via pypdf)]\n"
            f"• PDF URL: {top['website_pdf_url']}\n"
            f"• Total Pages: {top['total_pages']} (Best Match on Page {top['best_matching_page']})\n"
            f"• Extracted PDF Page Text:\n{top['pdf_page_excerpt']}"
        )
    else:
        summary = "[3. Website Policy PDF MCP Calls] No matching PDF pages found."

    return {"source_context": summary, "mcp_tool_calls": logs}


@mlflow_trace(name="langgraph.minio_mcp_node", span_type="AGENT")
def minio_mcp_node(state: QAState) -> Dict[str, Any]:
    """[4. MinIO Object Storage MCP Node] Calls `minio_list_objects`, `minio_query_csv`, `minio_read_object`."""
    logs = list(state.get("mcp_tool_calls", []))
    question = state["question"]
    q_lower = question.lower()

    _ = minio_list_objects()
    logs.append("MCP Call [MinIO S3]: minio_list_objects()")

    if any(k in q_lower for k in ["job", "scientist", "engineer", "bengaluru", "bangalore", "salary", "indian_tech_jobs"]):
        filter_kw = "Data Scientist Bengaluru" if "bengaluru" in q_lower or "bangalore" in q_lower else "Data Scientist"
        csv_json = minio_query_csv("business-data-dumps/indian_tech_jobs_2026.csv", filter_keyword=filter_kw, limit=3)
        logs.append(
            f"MCP Call [MinIO S3]: minio_query_csv('business-data-dumps/indian_tech_jobs_2026.csv', filter_keyword={filter_kw!r})"
        )
        csv_data = json.loads(csv_json)
        summary = (
            f"[4. MinIO Object Storage MCP Calls -> `s3://ai-source-data/{csv_data.get('object_key')}`]\n"
            f"• Total Rows in CSV: {csv_data.get('total_csv_rows')}\n"
            f"• Matching Rows for '{filter_kw}': {csv_data.get('matching_row_count')}\n"
            f"• Sample Rows:\n{json.dumps(csv_data.get('sample_matching_rows', []), indent=2)}"
        )
    else:
        md_json = minio_search_markdown_sources(question)
        logs.append(f"MCP Call [MinIO S3]: minio_search_markdown_sources({question!r})")
        md_data = json.loads(md_json)
        matches = md_data.get("top_matches", [])
        if matches:
            top = matches[0]
            _ = minio_read_object(top["object_key"], max_chars=2500)
            logs.append(f"MCP Call [MinIO S3]: minio_read_object({top['object_key']!r})")
            summary = (
                f"[4. MinIO Object Storage MCP Calls -> `s3://ai-source-data/{top['object_key']}`]\n"
                f"• Extracted Object Content:\n{top['matching_snippet']}"
            )
        else:
            summary = f"[4. MinIO Object Storage MCP Calls] Bucket `ai-source-data` inspected."

    return {"source_context": summary, "mcp_tool_calls": logs}


@mlflow_trace(name="langgraph.elastic_mcp_node", span_type="AGENT")
def elastic_mcp_node(state: QAState) -> Dict[str, Any]:
    """[5. Elasticsearch MCP Node] Search the Elasticsearch index built from MinIO, then read the hit back from MinIO."""
    logs = list(state.get("mcp_tool_calls", []))
    question = state["question"]

    _ = elastic_index_stats()
    logs.append("MCP Call [Elasticsearch]: elastic_index_stats()")

    search_data = json.loads(elastic_search_documents(question, size=3))
    logs.append(f"MCP Call [Elasticsearch]: elastic_search_documents({question!r})")

    hits = search_data.get("top_hits", [])
    if not hits:
        return {
            "source_context": "[5. Elasticsearch MCP Calls] No matching documents found in the index.",
            "mcp_tool_calls": logs,
        }

    top = hits[0]
    obj = json.loads(minio_read_object(top["minio_key"], max_chars=1500))
    logs.append(f"MCP Call [MinIO S3]: minio_read_object({top['minio_key']!r})  <- key returned by Elasticsearch")

    other = "\n".join(f"    {i}. {h['minio_key']} (score {h['score']})" for i, h in enumerate(hits[1:], start=2))
    highlight = " ... ".join(top["highlights"]) if top["highlights"] else top["excerpt"][:500]
    summary = (
        f"[5. Elasticsearch MCP Calls -> index `minio-documents` over MinIO `ai-source-data`]\n"
        f"• Matching files: {search_data.get('total_matching_files')} (company filter: {search_data['filters']['company'] or 'none'})\n"
        f"• Best hit (score {top['score']}): `{top['minio_url']}`"
        f"{' page ' + str(top['page']) if top['page'] else ''}\n"
        f"• Matched text (»term« = Elasticsearch highlight):\n{highlight}\n"
        f"• Other hits:\n{other or '    (none)'}\n"
        f"• Original read back from MinIO ({obj.get('total_length')} chars total), first 400 chars:\n"
        f"{obj.get('content', '')[:400]}"
    )
    return {"source_context": summary, "mcp_tool_calls": logs}


@mlflow_trace(name="langgraph.synthesizer_node", span_type="CHAIN")
def synthesizer_node(state: QAState) -> Dict[str, Any]:
    """Synthesizes the final answer from the selected MCP Node output."""
    question = state["question"]
    ctx = state.get("source_context", "")

    llm_ans = _call_llm_if_available(
        "Answer the user's question clearly using ONLY the MCP tool output below and cite the source.",
        f"Question: {question}\n\n{ctx}",
    )
    return {"final_answer": llm_ans if llm_ans else ctx}


# =====================================================================
# 4. BUILD & EXECUTE THE MULTI-NODE LANGGRAPH WORKFLOW
# =====================================================================

@mlflow_trace(name="LangGraph_MultiSource_MCP_Pipeline", span_type="AGENT")
def _execute_langgraph_pipeline(question: str) -> QAState:
    initial_state: QAState = {
        "question": question,
        "route": "postgres",
        "mcp_tool_calls": [],
        "source_context": "",
        "final_answer": "",
    }

    if HAS_LANGGRAPH:
        workflow = StateGraph(QAState)
        workflow.add_node("router", router_node)
        workflow.add_node("postgres_mcp", postgres_mcp_node)
        workflow.add_node("website_mcp", website_mcp_node)
        workflow.add_node("pdf_mcp", pdf_mcp_node)
        workflow.add_node("minio_mcp", minio_mcp_node)
        workflow.add_node("elastic_mcp", elastic_mcp_node)
        workflow.add_node("synthesizer", synthesizer_node)

        workflow.add_edge(START, "router")
        workflow.add_conditional_edges(
            "router",
            lambda s: s["route"],
            {
                "postgres": "postgres_mcp",
                "website": "website_mcp",
                "pdf": "pdf_mcp",
                "minio": "minio_mcp",
                "elastic": "elastic_mcp",
            },
        )
        workflow.add_edge("postgres_mcp", "synthesizer")
        workflow.add_edge("website_mcp", "synthesizer")
        workflow.add_edge("pdf_mcp", "synthesizer")
        workflow.add_edge("minio_mcp", "synthesizer")
        workflow.add_edge("elastic_mcp", "synthesizer")
        workflow.add_edge("synthesizer", END)

        app = workflow.compile()
        return app.invoke(initial_state)

    state = dict(initial_state)
    state.update(router_node(state))
    node_map = {
        "postgres": postgres_mcp_node,
        "website": website_mcp_node,
        "pdf": pdf_mcp_node,
        "minio": minio_mcp_node,
        "elastic": elastic_mcp_node,
    }
    state.update(node_map[state["route"]](state))
    state.update(synthesizer_node(state))
    return state  # type: ignore


def run_langgraph_qa(question: str) -> QAState:
    start_ts = time.perf_counter()
    if HAS_MLFLOW:
        with mlflow.start_run(run_name=f"LangGraph [{question[:38]}]"):
            mlflow.set_tag("framework", "LangGraph")
            mlflow.set_tag("protocol", "MCP")
            mlflow.log_param("framework", "LangGraph")
            mlflow.log_param("question", question)

            result = _execute_langgraph_pipeline(question)
            elapsed_ms = round((time.perf_counter() - start_ts) * 1000, 2)

            mlflow.set_tag("mcp_source", result["route"])
            mlflow.log_param("mcp_route", result["route"])
            mlflow.log_metric("execution_time_ms", elapsed_ms)
            mlflow.log_metric("mcp_steps_count", len(result["mcp_tool_calls"]))
            mlflow.log_text(result["final_answer"], "final_answer.txt")
            mlflow.log_dict(
                {
                    "question": question,
                    "mcp_route": result["route"],
                    "mcp_tool_calls": result["mcp_tool_calls"],
                    "final_answer": result["final_answer"],
                },
                "langgraph_execution_trace.json",
            )
            return result

    return _execute_langgraph_pipeline(question)


def main():
    parser = argparse.ArgumentParser(description="LangGraph Q&A with Separate MCP Calls (PostgreSQL, Website, PDFs, MinIO).")
    parser.add_argument("--question", "-q", help="Single question to ask.")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive prompt to type questions.")
    args = parser.parse_args()

    if HAS_MLFLOW:
        print(f"[MLflow Tracking Enabled] URI: {MLFLOW_TRACKING_URI} | Experiment: {MLFLOW_EXPERIMENT_NAME}")

    if args.interactive:
        print("=" * 80)
        print("LangGraph + Separate MCP Calls Interactive Q&A (PostgreSQL | Website | PDFs | MinIO)")
        print("Type your question and press Enter (or type 'exit' to quit).")
        print("=" * 80)
        while True:
            try:
                user_q = input("\nAsk a question > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if not user_q or user_q.lower() in ("exit", "quit", "q"):
                print("Exiting.")
                break
            result = run_langgraph_qa(user_q)
            print("-" * 80)
            print("LangGraph + Separate MCP Trace:")
            for step in result["mcp_tool_calls"]:
                print(f"  -> {step}")
            print("-" * 80)
            print("ANSWER:")
            print(result["final_answer"])
            print("=" * 80)
        return

    # Default demo runs 1 question per separate MCP source (PostgreSQL, Website HTML, Website PDF, MinIO CSV)
    questions = (
        [args.question]
        if args.question
        else [
            "[PostgreSQL] How many orders exist per status in PostgreSQL and what is their total amount?",
            "[Website HTML] From the Amerimed Wellness Institute website FAQ page, how much does a video urgent care visit cost?",
            "[Website PDF] In the OceanX Logistics PDF documents, what is the policy on demurrage and detention free time?",
            "[MinIO CSV] In the MinIO bucket indian_tech_jobs_2026.csv, how many Data Scientist jobs are in Bengaluru?",
            "[Documents -> Elasticsearch, chosen automatically] What is OceanX's policy on dangerous goods and hazardous cargo?",
        ]
    )

    for idx, q in enumerate(questions, start=1):
        print("=" * 80)
        print(f"QUESTION {idx}: {q}")
        print("-" * 80)
        result = run_langgraph_qa(q)
        print("LangGraph + Separate MCP Trace:")
        for step in result["mcp_tool_calls"]:
            print(f"  -> {step}")
        print("-" * 80)
        print("ANSWER:")
        print(result["final_answer"])
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
