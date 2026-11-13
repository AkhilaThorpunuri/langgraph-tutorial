"""
MCP (Model Context Protocol) Server for Tinitiate AI Setup (with MLflow Tracing).

Provides SEPARATE, dedicated MCP Tool Calls for each service/source:
  1. PostgreSQL Database (`tinitiate-ai-postgres` on localhost:5432):
     - `postgres_list_tables()`
     - `postgres_describe_table(table_name)`
     - `postgres_run_query(sql)`
  2. Company Websites HTTP Server (`tinitiate-ai-websites` Nginx on http://localhost:8080):
     - `website_list_pages(company)`
     - `website_fetch_page(company, page)`
     - `website_search_pages(query, company)`
  3. Policy PDFs on Website / MinIO (`*.pdf` documents parsed with `pypdf`):
     - `pdf_list_documents(company)`
     - `pdf_read_document(pdf_url_or_key, max_chars)`
     - `pdf_search_documents(query, company)`
  4. MinIO Object Storage & CSV Data Dumps (`tinitiate-ai-minio` on http://localhost:9002):
     - `minio_list_objects(prefix)`
     - `minio_query_csv(object_key, filter_keyword, limit)`
     - `minio_read_object(object_key, max_chars)`
  5. Elasticsearch search index over the MinIO documents (`tinitiate-ai-elasticsearch` on http://localhost:9200):
     - `elastic_index_minio_documents(prefix, recreate)`  (MinIO -> Elasticsearch)
     - `elastic_search_documents(query, company, file_type, size)`  (search -> MinIO keys)
     - `elastic_index_stats()`

Usage:
  - Standalone MCP Server (stdio):
      python qa_mcp_agents/mcp_server.py
  - Imported by LangGraph (`ask_langgraph.py`) and CrewAI (`ask_crewai.py`).
"""

import csv
import io
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

import boto3
import psycopg2
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

try:
    from mcp.server.fastmcp import FastMCP
    HAS_FASTMCP = True
except ImportError:
    HAS_FASTMCP = False
    FastMCP = None

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    from elasticsearch import Elasticsearch, helpers as es_helpers
    HAS_ELASTIC = True
except ImportError:
    HAS_ELASTIC = False
    Elasticsearch = None  # type: ignore
    es_helpers = None  # type: ignore

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    mlflow = None  # type: ignore

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MLFLOW_URI = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db').replace(os.sep, '/')}"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_URI)
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "Tinitiate-MCP-PostgreSQL-MinIO-QA")


def configure_mlflow():
    if HAS_MLFLOW:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def mlflow_trace(name: str, span_type: str = "TOOL") -> Callable:
    """Wraps a function in an MLflow Trace span when mlflow is installed."""
    def decorator(fn: Callable) -> Callable:
        if HAS_MLFLOW and hasattr(mlflow, "trace"):
            return mlflow.trace(name=name, span_type=span_type)(fn)
        return fn
    return decorator


# Service Endpoints (matching tinitiate-ai-onboarding.yml defaults)
PG_HOST = os.environ.get("POSTGRES_HOST", "localhost")
PG_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
PG_USER = os.environ.get("POSTGRES_USER", "ai_admin")
PG_PASS = os.environ.get("POSTGRES_PASSWORD", "ai_admin_pw")
PG_DB = os.environ.get("POSTGRES_DB", "ai_setup")

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://localhost:9002")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ROOT_USER", "ai_admin")
MINIO_SECRET_KEY = os.environ.get("MINIO_ROOT_PASSWORD", "ai_admin_pw")
MINIO_BUCKET = os.environ.get("SEED_BUCKET", "ai-source-data")

ELASTIC_URL = os.environ.get("ELASTIC_URL", f"http://localhost:{os.environ.get('ELASTIC_PORT', '9200')}")
ELASTIC_USER = os.environ.get("ELASTIC_USER", "elastic")
ELASTIC_PASS = os.environ.get("ELASTIC_PASSWORD", "ai_admin_pw")
ELASTIC_INDEX = os.environ.get("ELASTIC_INDEX", "minio-documents")

WEBSITES_BASE_URL = os.environ.get(
    "WEBSITES_BASE_URL",
    f"http://localhost:{os.environ.get('WEBSITES_PORT', '8080')}/business-website-policy-documents",
)

COMPANIES = [
    "oceanx-logistics",
    "atlas-retail-bank",
    "amerimed-wellness-institute",
]
WEBSITE_HTML_PAGES = [
    "index.html",
    "about.html",
    "services.html",
    "policies.html",
    "faq.html",
    "careers.html",
    "contact.html",
]

if HAS_FASTMCP:
    mcp = FastMCP("tinitiate-ai-multi-source-mcp")
else:
    class _DummyMCP:
        def tool(self):
            def decorator(fn):
                return fn
            return decorator
        def run(self):
            print("Please install `mcp` (`pip install mcp`) to run as a stdio server.")
    mcp = _DummyMCP()


def _get_pg_conn():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASS,
        dbname=PG_DB,
    )


def _get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name="us-east-1",
    )


def _get_es_client():
    if not HAS_ELASTIC:
        raise RuntimeError("elasticsearch client not installed (`pip install \"elasticsearch>=8,<9\"`).")
    return Elasticsearch(ELASTIC_URL, basic_auth=(ELASTIC_USER, ELASTIC_PASS), request_timeout=30)


def _extract_keywords(query: str) -> List[str]:
    stopwords = {
        "what", "is", "the", "for", "in", "of", "and", "to", "how", "does", "do",
        "are", "a", "an", "on", "with", "by", "from", "at", "about", "tell", "me",
        "policy", "company", "document", "documents", "website", "page", "pdf", "pdfs",
        "minio", "postgres", "postgresql", "database", "db",
    }
    kws = [
        w.lower()
        for w in re.findall(r"[a-zA-Z0-9]+", query)
        if w.lower() not in stopwords and len(w) > 1
    ]
    return kws or [w.lower() for w in re.findall(r"[a-zA-Z0-9]+", query)]


# =====================================================================
# 1. POSTGRESQL MCP CALLS (`tinitiate-ai-postgres`)
# =====================================================================

@mcp.tool()
@mlflow_trace(name="mcp.postgres.postgres_list_tables", span_type="TOOL")
def postgres_list_tables() -> str:
    """
    [PostgreSQL MCP Call] List all SQL tables in the PostgreSQL database (`ai_setup`) with row counts.
    """
    conn = _get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                ORDER BY table_name;
                """
            )
            tables = [r[0] for r in cur.fetchall()]
            summary = []
            for tbl in tables:
                cur.execute(f'SELECT COUNT(*) FROM "{tbl}";')
                count = cur.fetchone()[0]
                summary.append({"table_name": tbl, "row_count": count})
            return json.dumps({"source": "PostgreSQL (tinitiate-ai-postgres)", "tables": summary}, indent=2)
    finally:
        conn.close()


@mcp.tool()
@mlflow_trace(name="mcp.postgres.postgres_describe_table", span_type="TOOL")
def postgres_describe_table(table_name: str) -> str:
    """
    [PostgreSQL MCP Call] Describe columns, data types, and 3 sample rows of a PostgreSQL table.
    """
    if not re.match(r"^[a-zA-Z0-9_]+$", table_name):
        return json.dumps({"error": "Invalid table name"})

    conn = _get_pg_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position;
                """,
                (table_name,),
            )
            columns = [dict(r) for r in cur.fetchall()]
            if not columns:
                return json.dumps({"error": f"Table '{table_name}' not found."})

            cur.execute(f'SELECT * FROM "{table_name}" LIMIT 3;')
            sample_rows = [dict(r) for r in cur.fetchall()]
            return json.dumps(
                {
                    "source": "PostgreSQL (tinitiate-ai-postgres)",
                    "table_name": table_name,
                    "columns": columns,
                    "sample_rows": sample_rows,
                },
                default=str,
                indent=2,
            )
    finally:
        conn.close()


@mcp.tool()
@mlflow_trace(name="mcp.postgres.postgres_run_query", span_type="TOOL")
def postgres_run_query(sql: str) -> str:
    """
    [PostgreSQL MCP Call] Execute a read-only SELECT query on PostgreSQL (`ai_setup`) and return rows as JSON.
    """
    cleaned = sql.strip()
    if not cleaned.upper().startswith(("SELECT", "WITH")):
        return json.dumps({"error": "Only read-only SELECT / WITH queries are permitted."})

    conn = _get_pg_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(cleaned)
            rows = [dict(r) for r in cur.fetchmany(100)]
            return json.dumps(
                {
                    "source": "PostgreSQL (tinitiate-ai-postgres)",
                    "sql": cleaned,
                    "row_count": len(rows),
                    "rows": rows,
                },
                default=str,
                indent=2,
            )
    except Exception as exc:
        return json.dumps({"error": str(exc), "sql": cleaned}, indent=2)
    finally:
        conn.close()


# =====================================================================
# 2. WEBSITE HTTP MCP CALLS (`tinitiate-ai-websites` on localhost:8080)
# =====================================================================

@mcp.tool()
@mlflow_trace(name="mcp.website.website_list_pages", span_type="TOOL")
def website_list_pages(company: str = "") -> str:
    """
    [Website MCP Call] List all live HTML web pages hosted on `http://localhost:8080/business-website-policy-documents/`.
    """
    target_companies = [company] if company in COMPANIES else COMPANIES
    pages = [{"portal_home": f"{WEBSITES_BASE_URL}/index.html"}]
    for comp in target_companies:
        for page in WEBSITE_HTML_PAGES:
            pages.append(
                {
                    "company": comp,
                    "page": page,
                    "url": f"{WEBSITES_BASE_URL}/{comp}/website/{page}",
                }
            )
    return json.dumps(
        {
            "source": "Company Websites Container (tinitiate-ai-websites HTTP)",
            "total_pages": len(pages),
            "pages": pages,
        },
        indent=2,
    )


@mcp.tool()
@mlflow_trace(name="mcp.website.website_fetch_page", span_type="TOOL")
def website_fetch_page(company: str, page: str = "index.html") -> str:
    """
    [Website MCP Call] Fetch a live HTML page over HTTP from `http://localhost:8080/business-website-policy-documents/<company>/website/<page>`
    and extract its title, headings, text content, and links.
    """
    url = (
        f"{WEBSITES_BASE_URL}/index.html"
        if company in ("", "portal", "root")
        else f"{WEBSITES_BASE_URL}/{company}/website/{page}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        return json.dumps({"error": f"HTTP request to {url} failed: {exc}"})

    soup = BeautifulSoup(resp.text, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    main_el = soup.find("main") or soup.body or soup
    headings = [h.get_text(" ", strip=True) for h in main_el.find_all(["h1", "h2", "h3"])]
    text_content = main_el.get_text("\n", strip=True)
    pdf_links = [
        a["href"]
        for a in main_el.find_all("a", href=True)
        if a["href"].lower().endswith(".pdf")
    ]

    return json.dumps(
        {
            "source": "Company Websites Container (tinitiate-ai-websites HTTP)",
            "url": url,
            "http_status": resp.status_code,
            "title": title,
            "headings": headings,
            "pdf_links_on_page": pdf_links,
            "page_text": text_content[:4000],
        },
        indent=2,
    )


@mcp.tool()
@mlflow_trace(name="mcp.website.website_search_pages", span_type="RETRIEVER")
def website_search_pages(query: str, company: str = "") -> str:
    """
    [Website MCP Call] Search across live HTML company website pages on `http://localhost:8080`
    (Home, About, Services, Policies, FAQ, Careers, Contact) for matching content.
    """
    keywords = _extract_keywords(query)
    q_lower = query.lower()

    target_companies = [company] if company in COMPANIES else [
        c for c in COMPANIES if c.split("-")[0] in q_lower
    ] or COMPANIES

    results = []
    for comp in target_companies:
        for page in WEBSITE_HTML_PAGES:
            url = f"{WEBSITES_BASE_URL}/{comp}/website/{page}"
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code != 200:
                    continue
            except Exception:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.title.get_text(strip=True) if soup.title else page
            main_el = soup.find("main") or soup.body or soup
            text = main_el.get_text("\n", strip=True)
            text_lower = text.lower()

            score = sum(2 for kw in keywords if kw in text_lower) + sum(
                3 for kw in keywords if kw in page.lower() or kw in title.lower()
            )
            if score > 0:
                blocks = [
                    b.get_text(" ", strip=True)
                    for b in main_el.find_all(["section", "details", "div", "tr", "li"])
                    if len(b.get_text(" ", strip=True)) > 30
                ]
                best_block = text[:900]
                best_b_score = -1
                for blk in blocks:
                    b_score = sum(1 for kw in keywords if kw in blk.lower())
                    if b_score > best_b_score and len(blk) <= 1200:
                        best_b_score = b_score
                        best_block = blk

                results.append(
                    {
                        "company": comp,
                        "page": page,
                        "url": url,
                        "title": title,
                        "score": score,
                        "matching_excerpt": best_block[:1000],
                    }
                )

    results.sort(key=lambda x: x["score"], reverse=True)
    return json.dumps(
        {
            "source": "Company Websites Container (tinitiate-ai-websites HTTP)",
            "query": query,
            "keywords": keywords,
            "top_matching_pages": results[:5],
        },
        indent=2,
    )


# =====================================================================
# 3. WEBSITE POLICY PDF MCP CALLS (`*.pdf` via HTTP / `pypdf`)
# =====================================================================

@mcp.tool()
@mlflow_trace(name="mcp.pdf.pdf_list_documents", span_type="TOOL")
def pdf_list_documents(company: str = "") -> str:
    """
    [PDF MCP Call] List all Policy PDF documents (`*.pdf`) hosted on the company websites
    (`http://localhost:8080/business-website-policy-documents/<company>/documents/*.pdf`).
    """
    s3 = _get_s3_client()
    prefix = (
        f"business-website-policy-documents/{company}/documents/"
        if company in COMPANIES
        else "business-website-policy-documents/"
    )
    paginator = s3.get_paginator("list_objects_v2")
    pdfs = []
    for page in paginator.paginate(Bucket=MINIO_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".pdf"):
                rel_path = key.removeprefix("business-website-policy-documents/")
                pdfs.append(
                    {
                        "pdf_name": os.path.basename(key),
                        "website_pdf_url": f"{WEBSITES_BASE_URL}/{rel_path}",
                        "minio_key": key,
                        "size_bytes": obj["Size"],
                    }
                )
    return json.dumps(
        {
            "source": "Website Policy PDFs (HTTP + pypdf)",
            "total_pdfs": len(pdfs),
            "pdfs": pdfs,
        },
        indent=2,
    )


@mcp.tool()
@mlflow_trace(name="mcp.pdf.pdf_read_document", span_type="TOOL")
def pdf_read_document(pdf_url_or_key: str, max_chars: int = 5000) -> str:
    """
    [PDF MCP Call] Download and extract text from a Policy PDF document (`*.pdf`) via HTTP
    from `http://localhost:8080/business-website-policy-documents/...` (or by PDF filename/key) using `pypdf`.
    """
    if not HAS_PYPDF:
        return json.dumps({"error": "pypdf is required to parse PDF documents."})

    if pdf_url_or_key.startswith(("http://", "https://")):
        url = pdf_url_or_key
    elif pdf_url_or_key.startswith("business-website-policy-documents/"):
        rel = pdf_url_or_key.removeprefix("business-website-policy-documents/")
        url = f"{WEBSITES_BASE_URL}/{rel}"
    else:
        url = f"{WEBSITES_BASE_URL}/{pdf_url_or_key.lstrip('/')}"

    raw_bytes = None
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            raw_bytes = resp.content
    except Exception:
        pass

    if raw_bytes is None:
        # Fallback to MinIO bucket if a raw MinIO key was passed
        s3 = _get_s3_client()
        key = pdf_url_or_key if pdf_url_or_key.startswith("business-website-") or pdf_url_or_key.startswith("file-formats/") else f"business-website-policy-documents/{pdf_url_or_key}"
        resp_s3 = s3.get_object(Bucket=MINIO_BUCKET, Key=key)
        raw_bytes = resp_s3["Body"].read()

    reader = PdfReader(io.BytesIO(raw_bytes))
    pages_text = []
    for idx, page in enumerate(reader.pages, start=1):
        pages_text.append(f"[Page {idx}]\n{(page.extract_text() or '').strip()}")
    full_text = "\n\n".join(pages_text)

    return json.dumps(
        {
            "source": "Website Policy PDF (parsed with pypdf)",
            "pdf_url": url,
            "page_count": len(reader.pages),
            "extracted_text": full_text[:max_chars],
        },
        indent=2,
    )


@mcp.tool()
@mlflow_trace(name="mcp.pdf.pdf_search_documents", span_type="RETRIEVER")
def pdf_search_documents(query: str, company: str = "") -> str:
    """
    [PDF MCP Call] Search inside the actual Policy PDF files (`*.pdf`) hosted on `http://localhost:8080`
    using `pypdf` and return the matching PDF URL, page count, and extracted PDF text snippet.
    """
    keywords = _extract_keywords(query)
    q_lower = query.lower()

    if not company:
        for comp in COMPANIES:
            if comp.split("-")[0] in q_lower:
                company = comp
                break

    pdf_list_data = json.loads(pdf_list_documents(company))
    pdfs = pdf_list_data.get("pdfs", [])

    # First rank candidate PDFs by filename match so we don't have to parse all 27 PDFs unnecessarily
    scored_candidates = []
    for item in pdfs:
        fname_lower = item["pdf_name"].lower()
        name_score = sum(3 for kw in keywords if kw in fname_lower)
        scored_candidates.append((name_score, item))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    # Inspect top 8 candidate PDFs (or all if company-specific)
    candidates_to_parse = [item for _, item in scored_candidates[:9]]

    matches = []
    for item in candidates_to_parse:
        pdf_url = item["website_pdf_url"]
        try:
            resp = requests.get(pdf_url, timeout=8)
            if resp.status_code != 200:
                continue
            reader = PdfReader(io.BytesIO(resp.content))
            pages = [p.extract_text() or "" for p in reader.pages]
            full_text = "\n".join(pages)
        except Exception:
            continue

        full_lower = full_text.lower()
        score = sum(2 for kw in keywords if kw in full_lower) + sum(
            4 for kw in keywords if kw in item["pdf_name"].lower()
        )
        if score > 0:
            best_page_num = 1
            best_page_text = pages[0] if pages else ""
            best_p_score = -1
            for idx, p_txt in enumerate(pages, start=1):
                p_score = sum(1 for kw in keywords if kw in p_txt.lower())
                if p_score > best_p_score:
                    best_p_score = p_score
                    best_page_num = idx
                    best_page_text = p_txt

            matches.append(
                {
                    "pdf_name": item["pdf_name"],
                    "website_pdf_url": pdf_url,
                    "minio_key": item["minio_key"],
                    "total_pages": len(pages),
                    "best_matching_page": best_page_num,
                    "score": score,
                    "pdf_page_excerpt": best_page_text.strip()[:1200],
                }
            )

    matches.sort(key=lambda x: x["score"], reverse=True)
    return json.dumps(
        {
            "source": "Website Policy PDFs (HTTP + pypdf)",
            "query": query,
            "keywords": keywords,
            "top_pdf_matches": matches[:4],
        },
        indent=2,
    )


# =====================================================================
# 4. MINIO OBJECT STORAGE & CSV MCP CALLS (`tinitiate-ai-minio`)
# =====================================================================

@mcp.tool()
@mlflow_trace(name="mcp.minio.minio_list_objects", span_type="TOOL")
def minio_list_objects(prefix: str = "") -> str:
    """
    [MinIO MCP Call] List objects in the MinIO `ai-source-data` bucket under the given prefix.
    """
    s3 = _get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    files: List[Dict[str, Any]] = []
    for page in paginator.paginate(Bucket=MINIO_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            files.append({"key": obj["Key"], "size_bytes": obj["Size"]})
    return json.dumps(
        {
            "source": f"MinIO Object Storage (s3://{MINIO_BUCKET})",
            "prefix": prefix,
            "total_objects": len(files),
            "objects": files[:100],
        },
        indent=2,
    )


@mcp.tool()
@mlflow_trace(name="mcp.minio.minio_query_csv", span_type="TOOL")
def minio_query_csv(
    object_key: str = "business-data-dumps/indian_tech_jobs_2026.csv",
    filter_keyword: str = "",
    limit: int = 5,
) -> str:
    """
    [MinIO MCP Call] Query a CSV file stored inside the MinIO `ai-source-data` bucket
    (e.g. `business-data-dumps/indian_tech_jobs_2026.csv` or `file-formats/csv/customers-100.csv`).
    Filters rows matching `filter_keyword` and returns total matching count + sample rows.
    """
    s3 = _get_s3_client()
    try:
        resp = s3.get_object(Bucket=MINIO_BUCKET, Key=object_key)
        raw_text = resp["Body"].read().decode("utf-8", errors="replace")
    except Exception as exc:
        return json.dumps({"error": f"Failed to read CSV '{object_key}' from MinIO: {exc}"})

    reader = csv.DictReader(io.StringIO(raw_text))
    keywords = [k.lower() for k in filter_keyword.split() if k.strip()] if filter_keyword else []

    total_rows = 0
    matching_rows = []
    total_matches = 0

    for row in reader:
        total_rows += 1
        if keywords:
            row_str = " ".join(str(v) for v in row.values()).lower()
            if not all(k in row_str for k in keywords):
                continue
        total_matches += 1
        if len(matching_rows) < limit:
            matching_rows.append(row)

    return json.dumps(
        {
            "source": f"MinIO Object Storage CSV (s3://{MINIO_BUCKET}/{object_key})",
            "object_key": object_key,
            "columns": reader.fieldnames,
            "total_csv_rows": total_rows,
            "filter_keyword": filter_keyword,
            "matching_row_count": total_matches,
            "sample_matching_rows": matching_rows,
        },
        indent=2,
    )


@mcp.tool()
@mlflow_trace(name="mcp.minio.minio_read_object", span_type="TOOL")
def minio_read_object(object_key: str, max_chars: int = 6000) -> str:
    """
    [MinIO MCP Call] Read raw text/Markdown/YAML/PDF content of an object from the MinIO `ai-source-data` bucket.
    """
    s3 = _get_s3_client()
    try:
        resp = s3.get_object(Bucket=MINIO_BUCKET, Key=object_key)
        raw_bytes = resp["Body"].read()
    except Exception as exc:
        return json.dumps({"error": f"Could not read '{object_key}' from MinIO: {exc}"})

    if object_key.lower().endswith(".pdf") and HAS_PYPDF:
        reader = PdfReader(io.BytesIO(raw_bytes))
        text = "\n".join(f"[Page {i+1}]\n{p.extract_text() or ''}" for i, p in enumerate(reader.pages))
    else:
        text = raw_bytes.decode("utf-8", errors="replace")

    return json.dumps(
        {
            "source": f"MinIO Object Storage (s3://{MINIO_BUCKET}/{object_key})",
            "object_key": object_key,
            "total_length": len(text),
            "content": text[:max_chars],
        },
        indent=2,
    )


@mcp.tool()
@mlflow_trace(name="mcp.minio.minio_search_markdown_sources", span_type="RETRIEVER")
def minio_search_markdown_sources(query: str, prefix: str = "business-website-policy-documents/") -> str:
    """
    [MinIO MCP Call] Search Markdown & YAML files (`.md`, `.yaml`) stored in the MinIO `ai-source-data` bucket.
    """
    s3 = _get_s3_client()
    keywords = _extract_keywords(query)
    paginator = s3.get_paginator("list_objects_v2")
    matches = []

    for page in paginator.paginate(Bucket=MINIO_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.lower().endswith((".md", ".yaml", ".yml")):
                continue
            body = s3.get_object(Bucket=MINIO_BUCKET, Key=key)["Body"].read().decode("utf-8", errors="replace")
            body_lower = body.lower()

            score = sum(1 for kw in keywords if kw in body_lower) + sum(
                2 for kw in keywords if kw in key.lower()
            )
            if score > 0:
                front_matter = ""
                body_content = body
                fm_match = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)$", body, flags=re.DOTALL)
                if fm_match:
                    front_matter = fm_match.group(1).strip()
                    body_content = fm_match.group(2).strip()

                sections = [s.strip() for s in re.split(r"\r?\n(?=##\s)", body_content) if s.strip()]
                if not sections:
                    sections = [p.strip() for p in re.split(r"\r?\n\s*\r?\n", body_content) if p.strip()]

                best_sec = ""
                best_sec_score = -1
                for sec in sections:
                    s_score = sum(1 for kw in keywords if kw in sec.lower())
                    if s_score > best_sec_score:
                        best_sec_score = s_score
                        best_sec = sec

                snippet = f"{best_sec[:1200]}" if best_sec else body[:1200]
                if front_matter:
                    title_lines = [
                        line for line in front_matter.splitlines()
                        if line.startswith(("doc_id:", "title:", "category:"))
                    ]
                    if title_lines:
                        snippet = "\n".join(title_lines) + "\n\n" + snippet

                matches.append(
                    {
                        "object_key": key,
                        "score": score,
                        "matching_snippet": snippet,
                    }
                )

    matches.sort(key=lambda x: x["score"], reverse=True)
    return json.dumps(
        {
            "source": f"MinIO Object Storage (s3://{MINIO_BUCKET})",
            "query": query,
            "keywords": keywords,
            "top_matches": matches[:5],
        },
        indent=2,
    )


# =====================================================================
# 5. ELASTICSEARCH MCP CALLS (`tinitiate-ai-elasticsearch`)
#    Data lives in MinIO; Elasticsearch is the full-text search index over it.
#    Flow: MinIO object -> text chunks -> Elasticsearch index -> search -> MinIO key -> read from MinIO
# =====================================================================

_ES_INDEXABLE_EXT = (".md", ".yaml", ".yml", ".pdf", ".html", ".txt")
_ES_MAPPING = {
    "properties": {
        "minio_key": {"type": "keyword"},
        "minio_url": {"type": "keyword"},
        "file_name": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
        "file_type": {"type": "keyword"},
        "company": {"type": "keyword"},
        "doc_id": {"type": "keyword"},
        "title": {"type": "text"},
        "chunk_no": {"type": "integer"},
        "page": {"type": "integer"},
        "content": {"type": "text", "analyzer": "english"},
        "indexed_at": {"type": "date"},
    }
}


def _company_from_key(key: str) -> str:
    parts = key.split("/")
    if len(parts) > 1 and parts[1] in COMPANIES:
        return parts[1]
    return ""


def _es_chunks_for_object(key: str, raw: bytes) -> List[Dict[str, Any]]:
    """Split one MinIO object into searchable chunks (PDF -> per page, md/yaml/html/txt -> per section)."""
    lower = key.lower()
    file_name = os.path.basename(key)
    base = {
        "minio_key": key,
        "minio_url": f"s3://{MINIO_BUCKET}/{key}",
        "file_name": file_name,
        "file_type": lower.rsplit(".", 1)[-1],
        "company": _company_from_key(key),
        "doc_id": "",
        "title": file_name,
    }
    chunks: List[Dict[str, Any]] = []

    if lower.endswith(".pdf"):
        if not HAS_PYPDF:
            return []
        reader = PdfReader(io.BytesIO(raw))
        for idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                chunks.append({**base, "page": idx, "content": text})
        if chunks:
            first_line = chunks[0]["content"].splitlines()
            doc_id_match = re.search(r"\b([A-Z]{2,4}-[A-Z]{2,4}-\d{3})\b", file_name)
            for c in chunks:
                c["doc_id"] = doc_id_match.group(1) if doc_id_match else ""
                c["title"] = file_name.rsplit(".", 1)[0]
        return chunks

    text = raw.decode("utf-8", errors="replace")
    if lower.endswith(".html"):
        soup = BeautifulSoup(text, "html.parser")
        if soup.title:
            base["title"] = soup.title.get_text(strip=True)
        text = (soup.find("main") or soup.body or soup).get_text("\n", strip=True)

    body = text
    fm = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)$", text, flags=re.DOTALL)
    if fm:
        for line in fm.group(1).splitlines():
            if line.startswith("doc_id:"):
                base["doc_id"] = line.split(":", 1)[1].strip().strip('"')
            elif line.startswith("title:"):
                base["title"] = line.split(":", 1)[1].strip().strip('"')
        body = fm.group(2)

    sections = [s.strip() for s in re.split(r"\r?\n(?=##\s)", body) if s.strip()]
    if not sections:
        sections = [body.strip()] if body.strip() else []
    for sec in sections:
        for start in range(0, len(sec), 4000):
            piece = sec[start:start + 4000].strip()
            if piece:
                chunks.append({**base, "page": 0, "content": piece})
    return chunks


def _es_ensure_index(es, recreate: bool = False) -> None:
    if recreate and es.indices.exists(index=ELASTIC_INDEX):
        es.indices.delete(index=ELASTIC_INDEX)
    if not es.indices.exists(index=ELASTIC_INDEX):
        es.indices.create(index=ELASTIC_INDEX, mappings=_ES_MAPPING)


@mcp.tool()
@mlflow_trace(name="mcp.elastic.elastic_index_minio_documents", span_type="TOOL")
def elastic_index_minio_documents(prefix: str = "", recreate: bool = False) -> str:
    """
    [Elasticsearch MCP Call] Read documents stored in MinIO (`.pdf`, `.md`, `.yaml`, `.html`, `.txt`),
    split them into chunks (PDF pages / Markdown sections) and index them into Elasticsearch
    (`minio-documents` index) so they can be found by full-text search. Safe to re-run (idempotent).
    """
    s3 = _get_s3_client()
    es = _get_es_client()
    _es_ensure_index(es, recreate=recreate)

    now = datetime.now(timezone.utc).isoformat()
    actions, files_indexed, files_skipped = [], 0, []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=MINIO_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if (
                not key.lower().endswith(_ES_INDEXABLE_EXT)
                or "/_tools/" in key
                or "__pycache__" in key
                or os.path.basename(key).lower() == "todo.md"  # internal notes, hidden from the sites too
            ):
                continue
            try:
                raw = s3.get_object(Bucket=MINIO_BUCKET, Key=key)["Body"].read()
                chunks = _es_chunks_for_object(key, raw)
            except Exception as exc:
                files_skipped.append({"key": key, "reason": str(exc)[:120]})
                continue
            if not chunks:
                files_skipped.append({"key": key, "reason": "no extractable text"})
                continue
            files_indexed += 1
            for n, chunk in enumerate(chunks):
                chunk["chunk_no"] = n
                chunk["indexed_at"] = now
                actions.append({"_index": ELASTIC_INDEX, "_id": f"{key}::{n}", "_source": chunk})

    ok, errors = es_helpers.bulk(es, actions, raise_on_error=False)
    es.indices.refresh(index=ELASTIC_INDEX)
    total = es.count(index=ELASTIC_INDEX)["count"]
    return json.dumps(
        {
            "source": f"Elasticsearch ({ELASTIC_URL}) index '{ELASTIC_INDEX}' <- MinIO s3://{MINIO_BUCKET}/{prefix}",
            "minio_files_indexed": files_indexed,
            "chunks_written": ok,
            "bulk_errors": len(errors) if isinstance(errors, list) else errors,
            "total_chunks_in_index": total,
            "skipped": files_skipped[:10],
        },
        indent=2,
    )


@mcp.tool()
@mlflow_trace(name="mcp.elastic.elastic_index_stats", span_type="TOOL")
def elastic_index_stats() -> str:
    """
    [Elasticsearch MCP Call] Show the Elasticsearch `minio-documents` index status: chunk count and
    breakdown by company and file type (shows what from MinIO is searchable).
    """
    es = _get_es_client()
    if not es.indices.exists(index=ELASTIC_INDEX):
        return json.dumps({"index": ELASTIC_INDEX, "exists": False, "hint": "Run elastic_index_minio_documents first."})
    agg = es.search(
        index=ELASTIC_INDEX,
        size=0,
        aggs={
            "by_company": {"terms": {"field": "company", "missing": "(none)", "size": 10}},
            "by_type": {"terms": {"field": "file_type", "size": 10}},
            "files": {"cardinality": {"field": "minio_key"}},
        },
    )
    a = agg["aggregations"]
    return json.dumps(
        {
            "source": f"Elasticsearch ({ELASTIC_URL})",
            "index": ELASTIC_INDEX,
            "exists": True,
            "total_chunks": agg["hits"]["total"]["value"],
            "distinct_minio_files": a["files"]["value"],
            "by_company": {b["key"]: b["doc_count"] for b in a["by_company"]["buckets"]},
            "by_file_type": {b["key"]: b["doc_count"] for b in a["by_type"]["buckets"]},
        },
        indent=2,
    )


def _es_clean_query(query: str, company: str = "") -> str:
    """Drop instruction/filler words ('use Elasticsearch to search the MinIO documents for ...') so only the
    real topic terms reach Elasticsearch. The company name is dropped too when it is already a filter."""
    filler = {
        "use", "using", "used", "elastic", "elasticsearch", "search", "searching", "find", "look", "lookup",
        "over", "minio", "documents", "document", "docs", "files", "file", "bucket", "index",
        "the", "a", "an", "in", "on", "at", "of", "for", "to", "from", "with", "by", "and", "or",
        "what", "which", "is", "are", "does", "do", "how", "me", "tell", "about", "please", "show", "give",
        "handle", "handles", "handled", "handling", "work", "works", "explain", "describe", "list", "get",
        "can", "could", "would", "should", "my", "our", "their", "there", "any", "this", "that", "it", "be",
        "when", "where", "why", "who", "i", "we", "you", "have", "has", "need", "want", "know",
    }
    drop = set(filler)
    if company:
        drop.update(company.split("-")[:1])
        if company == "atlas-retail-bank":
            drop.update({"bank", "atlas"})
    words = re.findall(r"[A-Za-z0-9]+", query)
    kept = [w for w in words if w.lower() not in drop]
    return " ".join(kept) if kept else query


@mcp.tool()
@mlflow_trace(name="mcp.elastic.elastic_search_documents", span_type="RETRIEVER")
def elastic_search_documents(query: str, company: str = "", file_type: str = "", size: int = 5) -> str:
    """
    [Elasticsearch MCP Call] Full-text search (relevance-ranked, fuzzy, with highlights) over the documents
    stored in MinIO. Each hit returns the MinIO object key so the original can be read back from MinIO.
    Optional filters: company (oceanx-logistics | atlas-retail-bank | amerimed-wellness-institute),
    file_type (pdf | md | yaml | html | txt).
    """
    es = _get_es_client()
    auto_indexed = None
    if not es.indices.exists(index=ELASTIC_INDEX) or es.count(index=ELASTIC_INDEX)["count"] == 0:
        auto_indexed = json.loads(elastic_index_minio_documents())

    q_lower = query.lower()
    if not company:
        company = next((c for c in COMPANIES if c.split("-")[0] in q_lower), "")
    clean_query = _es_clean_query(query, company)

    filters = []
    if company:
        filters.append({"term": {"company": company}})
    if file_type:
        filters.append({"term": {"file_type": file_type.lower().lstrip(".")}})

    resp = es.search(
        index=ELASTIC_INDEX,
        size=size,
        query={
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": clean_query,
                            "fields": ["content", "title^3", "file_name^2", "doc_id^3"],
                            "fuzziness": "AUTO",
                            "minimum_should_match": "40%",
                        }
                    }
                ],
                "should": [
                    {"match_phrase": {"content": {"query": clean_query, "slop": 4, "boost": 3}}},
                    {"match_phrase": {"title": {"query": clean_query, "slop": 2, "boost": 4}}},
                ],
                "filter": filters,
            }
        },
        collapse={"field": "minio_key"},
        highlight={
            "pre_tags": ["»"],
            "post_tags": ["«"],
            "fields": {"content": {"fragment_size": 350, "number_of_fragments": 2}},
        },
    )

    hits = []
    for h in resp["hits"]["hits"]:
        s = h["_source"]
        hits.append(
            {
                "score": round(h["_score"], 3),
                "minio_key": s["minio_key"],
                "minio_url": s["minio_url"],
                "title": s["title"],
                "doc_id": s.get("doc_id", ""),
                "company": s.get("company", ""),
                "file_type": s["file_type"],
                "page": s.get("page", 0),
                "highlights": h.get("highlight", {}).get("content", []),
                "excerpt": s["content"][:700],
            }
        )

    result: Dict[str, Any] = {
        "source": f"Elasticsearch ({ELASTIC_URL}) index '{ELASTIC_INDEX}' over MinIO s3://{MINIO_BUCKET}",
        "query": query,
        "query_sent_to_elasticsearch": clean_query,
        "filters": {"company": company, "file_type": file_type},
        "total_matching_files": resp["hits"]["total"]["value"],
        "top_hits": hits,
    }
    if auto_indexed:
        result["auto_indexed_first"] = {
            "minio_files_indexed": auto_indexed.get("minio_files_indexed"),
            "total_chunks_in_index": auto_indexed.get("total_chunks_in_index"),
        }
    return json.dumps(result, indent=2)


# Backward-compatible aliases
list_db_tables = postgres_list_tables
describe_db_table = postgres_describe_table
run_sql_query = postgres_run_query
list_minio_files = minio_list_objects
read_minio_document = minio_read_object
search_minio_documents = minio_search_markdown_sources


if __name__ == "__main__":
    print("Starting Tinitiate AI Multi-Source MCP Server (PostgreSQL + Website HTTP + Website PDFs + MinIO)...")
    mcp.run()
