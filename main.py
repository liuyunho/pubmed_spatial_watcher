# main.py
from __future__ import annotations

from datetime import datetime, date, timedelta
from typing import List, Optional
from datetime import date

import re

from fastapi import FastAPI, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    Date,
    Boolean,
    DateTime,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from Bio import Entrez, Medline

# ---------- CONFIG ----------
DATABASE_URL = "sqlite:///./papers.db"
ENTREZ_EMAIL = "your_email@example.com"  # NCBI requires an email
SEARCH_TERM = (
    '(spatial transcriptomics OR spatial proteomics OR "spatial omics" OR '
    'Visium OR Xenium OR CosMx OR GeoMx OR MERFISH OR seqFISH OR CODEX OR MIBI) '
    'AND (cancer OR tumor OR tissue OR pathology)'
)

# ---------- DB SETUP ----------
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Paper(Base):
    __tablename__ = "papers"
    id = Column(Integer, primary_key=True)
    pmid = Column(String, unique=True, index=True, nullable=False)
    title = Column(Text, nullable=False)
    abstract = Column(Text)
    journal = Column(String)
    pub_date = Column(Date)
    doi = Column(String)
    is_spatial = Column(Boolean, default=False)
    platforms = Column(String)   # e.g. "Visium; Xenium"
    sample_info = Column(String) # small JSON-ish string
    created_at = Column(DateTime, default=datetime.utcnow)


class Meta(Base):
    __tablename__ = "meta"
    key = Column(String, primary_key=True)
    value = Column(String)


Base.metadata.create_all(bind=engine)

# ---------- CLASSIFICATION LOGIC ----------
SPATIAL_KEYWORDS = [
    "spatial transcriptomics",
    "spatial proteomics",
    "spatial multi-omics",
    "spatial omics",
    "visium",
    "xenium",
    "cosmx",
    "geomx",
    "merfish",
    "seqfish",
    "codex",
    "mibi",
    "imaging mass cytometry",
    "nanostring",
]

PLATFORM_KEYWORDS = {
    "Visium": ["visium"],
    "Xenium": ["xenium"],
    "CosMx": ["cosmx"],
    "GeoMx": ["geomx"],
    "MERFISH": ["merfish"],
    "seqFISH": ["seqfish"],
    "CODEX/PhenoCycler": ["codex", "phenocycler"],
    "MIBI": ["mibi"],
}


def classify_paper(title: str, abstract: Optional[str]) -> tuple[bool, List[str]]:
    txt = (title or "") + " " + (abstract or "")
    txt_low = txt.lower()
    is_spatial = any(kw in txt_low for kw in SPATIAL_KEYWORDS)
    platforms: List[str] = []
    if is_spatial:
        for platform, kws in PLATFORM_KEYWORDS.items():
            if any(kw in txt_low for kw in kws):
                platforms.append(platform)
    return is_spatial, platforms


def extract_sample_info(text: str) -> str:
    """
    Very rough demo: look for 'n=XX patients' or 'XX patients'.
    You can later replace this with LLM-based extraction.
    """
    if not text:
        return ""
    # n=42 patients
    m = re.search(r"(?:n\s*=?\s*)(\d+)\s+(patients?|cases?)", text, re.IGNORECASE)
    if m:
        return f"n_patients={m.group(1)}"
    # 42 patients
    m2 = re.search(r"\b(\d+)\s+(patients?|cases?)", text, re.IGNORECASE)
    if m2:
        return f"n_patients={m2.group(1)}"
    return ""


# ---------- PUBMED CLIENT ----------
def fetch_pubmed_batch(
    query: str,
    since_date: date,
    email: str,
    max_results: int = 2000,
) -> List[dict]:
    Entrez.email = email
    today_str = datetime.today().strftime("%Y/%m/%d")
    since_str = since_date.strftime("%Y/%m/%d")

    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        datetype="pdat",
        mindate=since_str,
        maxdate=today_str,
        usehistory="y",      # enable WebEnv + QueryKey
        retmax=0,
    )
    search_record = Entrez.read(handle)
    count = int(search_record["Count"])
    webenv = search_record["WebEnv"]
    query_key = search_record["QueryKey"]

    records: List[dict] = []
    batch_size = 200
    for start in range(0, min(count, max_results), batch_size):
        h = Entrez.efetch(
            db="pubmed",
            rettype="medline",
            retmode="text",
            retstart=start,
            retmax=batch_size,
            webenv=webenv,
            query_key=query_key,
        )
        for r in Medline.parse(h):
            pmid = r.get("PMID")
            title = r.get("TI", "")
            abstract = r.get("AB", "")
            journal = r.get("JT", "")
            dp = r.get("DP", "")
            doi = None
            for id_ in r.get("AID", []):
                if id_.endswith(" [doi]"):
                    doi = id_.split(" ")[0]
                    break
            pub_date_val = None
            if len(dp) >= 4 and dp[:4].isdigit():
                year = int(dp[:4])
                pub_date_val = date(year, 1, 1)
            records.append(
                dict(
                    pmid=pmid,
                    title=title,
                    abstract=abstract,
                    journal=journal,
                    pub_date=pub_date_val,
                    doi=doi,
                )
            )
    return records


# ---------- DB UTILITIES ----------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_last_sync(db: Session) -> date:
    meta = db.get(Meta, "last_pubmed_sync")
    if meta is None:
        # FIRST RUN: start from 2020-01-01
        return date(2020, 1, 1)
    # LATER RUNS: continue from last saved date
    return date.fromisoformat(meta.value)



def set_last_sync(db: Session, d: date) -> None:
    meta = db.get(Meta, "last_pubmed_sync")
    if meta is None:
        meta = Meta(key="last_pubmed_sync", value=d.isoformat())
        db.add(meta)
    else:
        meta.value = d.isoformat()
    db.commit()


def upsert_papers(db: Session, records: List[dict]) -> int:
    """
    Insert new papers or update existing ones.
    Returns number of papers inserted/updated.
    """
    count = 0
    for rec in records:
        pmid = rec["pmid"]
        paper = db.query(Paper).filter_by(pmid=pmid).one_or_none()
        is_spatial, platforms = classify_paper(rec["title"], rec["abstract"])
        sample_info = extract_sample_info(rec["abstract"])

        if paper is None:
            paper = Paper(
                pmid=pmid,
                title=rec["title"],
                abstract=rec["abstract"],
                journal=rec["journal"],
                pub_date=rec["pub_date"],
                doi=rec["doi"],
                is_spatial=is_spatial,
                platforms="; ".join(platforms) if platforms else "",
                sample_info=sample_info,
            )
            db.add(paper)
            count += 1
        else:
            # Update basic fields if changed
            paper.title = rec["title"]
            paper.abstract = rec["abstract"]
            paper.journal = rec["journal"]
            paper.pub_date = rec["pub_date"]
            paper.doi = rec["doi"]
            paper.is_spatial = is_spatial
            paper.platforms = "; ".join(platforms) if platforms else ""
            paper.sample_info = sample_info
            count += 1
    db.commit()
    return count


def sync_pubmed(db: Session) -> int:
    """
    High-level update function:
    - read last sync date
    - fetch new records
    - upsert into DB
    - update last sync date
    """
    last = get_last_sync(db)
    records = fetch_pubmed_batch(
        query=SEARCH_TERM,
        since_date=last,
        email=ENTREZ_EMAIL,
        max_results=500,
    )
    n = upsert_papers(db, records)
    set_last_sync(db, date.today())
    return n


# ---------- FASTAPI APP + TEMPLATES ----------
app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.on_event("startup")
def startup_sync():
    """Run a sync when the server starts."""
    db = SessionLocal()
    try:
        n = sync_pubmed(db)
        print(f"[startup] Synced {n} PubMed records.")
    finally:
        db.close()


@app.get("/papers", response_class=HTMLResponse)
def list_papers(
    request: Request,
    spatial_only: bool = True,
    platform: Optional[str] = None,
    db: Session = Depends(get_db),
):
    q = db.query(Paper)
    if spatial_only:
        q = q.filter(Paper.is_spatial == True)  # noqa: E712
    if platform:
        q = q.filter(Paper.platforms.contains(platform))
    papers = q.order_by(Paper.pub_date.desc().nullslast(), Paper.id.desc()).limit(200).all()

    return templates.TemplateResponse(
        "papers.html",
        {
            "request": request,
            "papers": papers,
            "spatial_only": spatial_only,
            "platform": platform or "",
        },
    )


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    total = db.query(Paper).count()
    spatial = db.query(Paper).filter(Paper.is_spatial == True).count()  # noqa: E712
    latest = (
        db.query(Paper)
        .filter(Paper.is_spatial == True)  # noqa: E712
        .order_by(Paper.pub_date.desc().nullslast(), Paper.id.desc())
        .limit(10)
        .all()
    )
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "total": total,
            "spatial": spatial,
            "latest": latest,
        },
    )
