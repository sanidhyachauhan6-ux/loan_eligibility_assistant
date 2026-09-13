# ingest.py — ClaimAssist v3 RAG ingestion (Session 8)
#
# Reads data/policies/*.md and splits them BY CLAUSE, not by fixed-size window.
# The policy documents were written with numbered clause headings ("M-2.3
# Licence validity:", "H-4.2 Room rent capping:", "P-1.4 Surveyor timelines:")
# precisely so that one chunk == one citable unit. When the LLM later answers
# "your claim was rejected under [M-2.3]", that citation resolves to exactly
# one chunk — which is what makes citations verifiable in a regulated domain.
#
# Chunks are embedded with chromadb's DEFAULT embedding function
# (all-MiniLM-L6-v2 exported to ONNX, ~80 MB, downloaded on first run and
# cached under ~/.cache/chroma/onnx_models) and stored in a PERSISTENT
# collection at rag/chroma/ so the API container can query the same store.
#
# The script is IDEMPOTENT: it deletes and recreates the collection on every
# run, so re-running after editing a policy document never leaves stale chunks.
#
# Run from the session8_lab directory:   python rag/ingest.py
import re
import sys
from pathlib import Path

import chromadb

LAB_ROOT = Path(__file__).resolve().parent
POLICIES_DIR = LAB_ROOT
CHROMA_DIR = LAB_ROOT / "chroma"
COLLECTION = "eligibility"

# Clause headings look like "M-2.3 Licence validity: ..." at the start of a
# line: a letter prefix (M=motor, H=health, P=process), a dotted number, a
# short title, then a colon. Everything until the next clause heading or
# section heading ("## Section ...") belongs to that clause.
import re


# Matches:
# ## 1. Home Loan
# ## 2. Personal Loan
# ## 7. General Applicant Information
# and also the repository’s bold-wrapped style:
# **## 1. Home Loan**
# **### 1.1 Age and Working Years**
SECTION_RE = re.compile(
    r"^\*{0,4}##\s+(?P<section>\d+)\.\s+(?P<title>.+?)\*{0,4}\s*$",
    re.MULTILINE,
)

SUBSECTION_RE = re.compile(
    r"^\*{0,4}###\s+(?P<section>\d+\.\d+)\s+(?P<title>.+?)\*{0,4}\s*$",
    re.MULTILINE,
)

RULE_ID_RE = re.compile(
    r"\*{0,4}Rule ID:\*{0,4}\s*:?\s*\*{0,4}"
    r"(?P<id>[A-Z]+(?:-[A-Z0-9]+)*-\d+)",
    re.IGNORECASE,
)


def split_by_clause(text: str, doc_name: str) -> list[dict]:
    chunks = []

    section_matches = list(SECTION_RE.finditer(text))

    for section_index, section_match in enumerate(section_matches):

        section_start = section_match.start()

        section_end = (
            section_matches[section_index + 1].start()
            if section_index + 1 < len(section_matches)
            else len(text)
        )

        section_text = text[section_start:section_end].strip()

        section = section_match.group("section")
        title = section_match.group("title").strip()

        subsection_matches = list(
            SUBSECTION_RE.finditer(section_text)
        )

        # ---------------------------------------------------------
        # Section contains ### subsections
        # ---------------------------------------------------------

        if subsection_matches:

            for sub_index, sub_match in enumerate(subsection_matches):

                sub_start = sub_match.start()

                sub_end = (
                    subsection_matches[sub_index + 1].start()
                    if sub_index + 1 < len(subsection_matches)
                    else len(section_text)
                )

                clause_text = section_text[sub_start:sub_end].strip()

                rule_match = RULE_ID_RE.search(clause_text)

                chunks.append({
                    "doc": doc_name,
                    "section": section,
                    "title": title,
                    "subsection": sub_match.group("section"),
                    "subsection_title": sub_match.group("title").strip(),
                    "rule_id": (
                        rule_match.group("id").upper()
                        if rule_match
                        else None
                    ),
                    "text": " ".join(clause_text.split()),
                })

        # ---------------------------------------------------------
        # Section has no subsections
        # ---------------------------------------------------------

        else:

            rule_match = RULE_ID_RE.search(section_text)

            chunks.append({
                "doc": doc_name,
                "section": section,
                "title": title,
                "subsection": None,
                "subsection_title": None,
                "rule_id": (
                    rule_match.group("id").upper()
                    if rule_match
                    else None
                ),
                "text": " ".join(section_text.split()),
            })

    return chunks

def main() -> None:
    policy_files = sorted(POLICIES_DIR.glob("*.md"))

    if not policy_files:
        sys.exit(f"No policy documents found in {POLICIES_DIR}")

    all_chunks: list[dict] = []

    for path in policy_files:
        text = path.read_text(encoding="utf-8")

        # Normalize the policy file’s bold-wrapped headers so the parser can
        # recognize the same sections and subsections the markdown uses
        # throughout the repository.
        text = re.sub(r"(?m)^\*{1,4}##\s+(?P<section>\d+)\.\s+(?P<title>.+?)\*{1,4}\s*$",
                      r"## \g<section>. \g<title>", text)
        text = re.sub(r"(?m)^\*{1,4}###\s+(?P<section>\d+\.\d+)\s+(?P<title>.+?)\*{1,4}\s*$",
                      r"### \g<section> \g<title>", text)

        chunks = split_by_clause(text, path.name)

        print(f"{path.name}: {len(chunks)} clauses")
        all_chunks.extend(chunks)

    # PersistentClient writes the index to disk — the API process opens the
    # same directory read/write and queries it. In production this directory
    # becomes a vector database service (pgvector, a managed vector DB, ...).
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Idempotent: drop and rebuild. Never append blindly — stale chunks from a
    # previous version of a document are a silent correctness bug in RAG.
    try:
        client.delete_collection(COLLECTION)
        print(
            f"Deleted existing collection '{COLLECTION}' "
            "(idempotent rebuild)"
        )
    except Exception:
        pass

    # No embedding_function argument => Chroma's default embedding function
    # (ONNX all-MiniLM-L6-v2).
    collection = client.create_collection(
        COLLECTION,
        metadata={"hnsw:space": "l2"}
    )

    # ---------------------------------------------------------
    # Build unique IDs and rich metadata for each chunk.
    #
    # Example:
    #   HOME-AGE-001
    #   PERSONAL-CREDIT-001
    #   AUTO-INCOME-001
    #
    # Rule ID is preferred because it is explicitly defined in
    # the policy and gives us a stable citation/reference key.
    # ---------------------------------------------------------
    ids = []

    for i, c in enumerate(all_chunks):
        if c["rule_id"]:
            chunk_id = c["rule_id"]
        else:
            # Fallback for sections without a Rule ID.
            chunk_id = (
                f"{c['doc']}:"
                f"{c['section']}:"
                f"{c.get('subsection', 'root')}:"
                f"{i}"
            )

        # Ensure uniqueness even if a rule ID accidentally appears
        # more than once across policy documents.
        if chunk_id in ids:
            chunk_id = f"{chunk_id}-{i}"

        ids.append(chunk_id)

    metadatas = []

    for c in all_chunks:
        metadata = {
            "doc": c["doc"],
            "section": c["section"],
            "title": c["title"],
        }

        if c.get("subsection") is not None:
            metadata["subsection"] = c["subsection"]

        if c.get("subsection_title") is not None:
            metadata["subsection_title"] = c["subsection_title"]

        if c.get("rule_id") is not None:
            metadata["rule_id"] = c["rule_id"]

        metadatas.append(metadata)

    collection.add(
        ids=ids,
        documents=[c["text"] for c in all_chunks],
        metadatas=metadatas,
    )

    print(
        f"\nStored {collection.count()} clause chunks "
        f"in {CHROMA_DIR}/"
    )

    print("Sample chunks (id · loan type · criterion · rule id):")

    for chunk_id, c in zip(ids[:5], all_chunks[:5]):
        print(
            f"  {chunk_id:<24} · "
            f"{c['title']:<20} · "
            f"{c.get('subsection_title', 'General'):<35} · "
            f"{str(c['rule_id']):<24}"
        )


if __name__ == "__main__":
    main()
