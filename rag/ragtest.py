import argparse
import json
import os
import re
import unicodedata
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from rag import RAGApplication, _file_hash, _load_index_record, _save_index_record


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(mean(values))


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _canon_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("—", "-").replace("–", "-")
    text = re.sub(r"[-_/]+", " ", text)
    text = re.sub(r"[^\w\u4e00-\u9fff.\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _build_keyword_groups(case: dict[str, Any]) -> list[list[str]]:
    groups_raw = case.get("expected_answer_keyword_groups")
    if isinstance(groups_raw, list):
        groups: list[list[str]] = []
        for item in groups_raw:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    groups.append([cleaned])
                continue
            if isinstance(item, list):
                aliases = [str(v).strip() for v in item if str(v).strip()]
                if aliases:
                    groups.append(aliases)
        if groups:
            return groups

    primary = case.get("expected_answer_keywords") or []
    secondary = case.get("expected_answer_keywords_zh") or []
    groups = []
    for i, keyword in enumerate(primary):
        aliases = [str(keyword).strip()]
        if i < len(secondary):
            zh_item = secondary[i]
            if isinstance(zh_item, list):
                aliases.extend(str(v).strip() for v in zh_item if str(v).strip())
            else:
                zh_value = str(zh_item).strip()
                if zh_value:
                    aliases.append(zh_value)
        aliases = [a for a in aliases if a]
        if aliases:
            groups.append(aliases)
    return groups


def _keyword_hit(answer_canon: str, answer_compact: str, keyword: str) -> bool:
    keyword_canon = _canon_text(keyword)
    if not keyword_canon:
        return False
    if keyword_canon in answer_canon:
        return True
    keyword_compact = _compact_text(keyword_canon)
    return len(keyword_compact) >= 2 and keyword_compact in answer_compact


def _match_source(expected: str, retrieved_sources: list[str]) -> bool:
    exp = expected.strip().lower()
    exp_base = os.path.basename(exp)
    for src in retrieved_sources:
        src_norm = src.strip().lower()
        if src_norm == exp or os.path.basename(src_norm) == exp_base:
            return True
    return False


def _score_answer(answer: str, case: dict[str, Any]) -> tuple[float | None, dict[str, float]]:
    parts: dict[str, float] = {}
    answer_norm = _norm_text(answer)
    answer_canon = _canon_text(answer)
    answer_compact = _compact_text(answer_canon)

    keyword_groups = _build_keyword_groups(case)
    if keyword_groups:
        hit = 0
        for group in keyword_groups:
            if any(_keyword_hit(answer_canon, answer_compact, alias) for alias in group):
                hit += 1
        parts["keyword_coverage"] = hit / len(keyword_groups)

    exact = case.get("expected_answer_exact")
    if exact:
        parts["exact_match"] = 1.0 if answer_norm == _norm_text(str(exact)) else 0.0

    regex = case.get("expected_answer_regex")
    if regex:
        matched = re.search(str(regex), answer, flags=re.IGNORECASE | re.DOTALL) is not None
        parts["regex_match"] = 1.0 if matched else 0.0

    final_score = _safe_mean(list(parts.values()))
    return final_score, parts


def _load_cases(cases_path: Path) -> list[dict[str, Any]]:
    if not cases_path.exists():
        raise FileNotFoundError(
            f"Case file not found: {cases_path}. "
            f"You can start from rag/ragtest_cases.example.json"
        )

    data = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Case file must be a JSON array")

    cleaned: list[dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Case #{i + 1} must be an object")
        question = str(item.get("question", "")).strip()
        if not question:
            raise ValueError(f"Case #{i + 1} missing non-empty 'question'")
        case_id = str(item.get("id") or f"case_{i + 1}")
        copied = dict(item)
        copied["id"] = case_id
        copied["question"] = question
        cleaned.append(copied)
    return cleaned


def _ensure_indexed(rag: RAGApplication, rag_dir: Path, reindex: bool) -> dict[str, int]:
    papers_dir = rag_dir / "papers"
    sample_path = rag_dir / "sample.txt"

    candidates: list[str] = []
    if papers_dir.exists():
        candidates.extend(str(p) for p in sorted(papers_dir.glob("*.pdf")))
    if sample_path.exists():
        candidates.append(str(sample_path))

    if reindex:
        rag.clear_collection()
        index_record: dict[str, str] = {}
    else:
        index_record = _load_index_record()

    indexed = 0
    skipped = 0

    for file_path in candidates:
        current_hash = _file_hash(file_path)
        if not reindex and index_record.get(file_path) == current_hash:
            skipped += 1
            continue

        docs = rag.load_documents(file_path)
        chunks = rag.split_documents(docs)
        rag.create_vectorstore(chunks)
        index_record[file_path] = current_hash
        indexed += 1

    if reindex or indexed > 0:
        _save_index_record(index_record)

    return {"indexed": indexed, "skipped": skipped, "candidates": len(candidates)}


def _create_llm() -> ChatOpenAI:
    api_key = os.getenv("LONGCAT_API_KEY") or os.getenv("OPENAI_API_KEY") or "DUMMY_KEY"
    model = os.getenv("LONGCAT_MODEL", "gpt-5.2-codex")
    base_url = os.getenv("LONGCAT_BASE_URL", "http://localhost:11434")
    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv()

    db_url = os.getenv("DATABASE_URL")
    embeddings_key = os.getenv("SILICONFLOW_API_KEY")
    embeddings_url = os.getenv("SILICONFLOW_BASE_URL")

    if not db_url:
        raise ValueError("DATABASE_URL is required")
    if not embeddings_key:
        raise ValueError("SILICONFLOW_API_KEY is required")
    if not embeddings_url:
        raise ValueError("SILICONFLOW_BASE_URL is required")

    rag_dir = Path(__file__).resolve().parent
    llm = _create_llm()
    rag = RAGApplication(
        llm_client=llm,
        embeddings_key=embeddings_key,
        embeddings_url=embeddings_url,
        db_url=db_url,
    )

    if args.index_if_needed or args.reindex:
        index_stats = _ensure_indexed(rag, rag_dir, reindex=args.reindex)
    else:
        index_stats = {"indexed": 0, "skipped": 0, "candidates": 0}

    rag.setup_retriever(k=args.top_k)
    if not args.skip_answer:
        rag.create_chain()

    cases = _load_cases(Path(args.cases).resolve())

    per_case: list[dict[str, Any]] = []
    source_recall_values: list[float] = []
    source_hit_values: list[float] = []
    source_precision_values: list[float] = []
    source_top1_hit_values: list[float] = []
    source_mrr_values: list[float] = []
    answer_acc_values: list[float] = []

    for case in cases:
        question = case["question"]
        retrieved_docs = rag.hybrid_retrieve(
            question=question,
            k=args.top_k,
            fetch_k=args.fetch_k,
        )

        retrieved_sources = [str(doc.metadata.get("source_file", "unknown")) for doc in retrieved_docs]
        retrieved_set = list(dict.fromkeys(retrieved_sources))

        expected_sources = [str(s) for s in (case.get("expected_sources") or [])]
        matched_sources: list[str] = []
        if expected_sources:
            matched_sources = [s for s in expected_sources if _match_source(s, retrieved_set)]
            source_recall = len(matched_sources) / len(expected_sources)
            source_hit = 1.0 if matched_sources else 0.0

            matched_retrieved = [
                src
                for src in retrieved_set
                if any(_match_source(exp, [src]) for exp in expected_sources)
            ]
            source_precision_like = (
                len(matched_retrieved) / len(retrieved_set) if retrieved_set else 0.0
            )
            source_top1_hit = (
                1.0
                if retrieved_set and any(_match_source(exp, [retrieved_set[0]]) for exp in expected_sources)
                else 0.0
            )
            source_first_match_rank = None
            for rank, src in enumerate(retrieved_set, start=1):
                if any(_match_source(exp, [src]) for exp in expected_sources):
                    source_first_match_rank = rank
                    break
            source_mrr = 1.0 / source_first_match_rank if source_first_match_rank else 0.0

            source_recall_values.append(source_recall)
            source_hit_values.append(source_hit)
            source_precision_values.append(source_precision_like)
            source_top1_hit_values.append(source_top1_hit)
            source_mrr_values.append(source_mrr)
        else:
            source_recall = None
            source_hit = None
            source_precision_like = None
            source_top1_hit = None
            source_first_match_rank = None
            source_mrr = None

        answer_text = None
        answer_accuracy = None
        answer_parts: dict[str, float] = {}
        if not args.skip_answer:
            result = rag.query(question, docs=retrieved_docs)
            answer_text = str(result.get("answer", ""))
            answer_accuracy, answer_parts = _score_answer(answer_text, case)
            if answer_accuracy is not None:
                answer_acc_values.append(answer_accuracy)

        item = {
            "id": case["id"],
            "question": question,
            "expected_sources": expected_sources,
            "matched_sources": matched_sources,
            "retrieved_sources": retrieved_set,
            "source_recall": source_recall,
            "source_hit": source_hit,
            "source_precision_like": source_precision_like,
            "source_top1_hit": source_top1_hit,
            "source_first_match_rank": source_first_match_rank,
            "source_mrr": source_mrr,
            "answer_accuracy": answer_accuracy,
            "answer_score_parts": answer_parts,
        }
        if args.include_answer_text:
            item["answer"] = answer_text
        per_case.append(item)

    summary = {
        "cases_total": len(cases),
        "source_labeled_cases": len(source_recall_values),
        "answer_labeled_cases": len(answer_acc_values),
        "avg_source_recall": _safe_mean(source_recall_values),
        "source_hit_rate": _safe_mean(source_hit_values),
        "avg_source_precision_like": _safe_mean(source_precision_values),
        "top1_source_hit_rate": _safe_mean(source_top1_hit_values),
        "source_mrr": _safe_mean(source_mrr_values),
        "avg_answer_accuracy": _safe_mean(answer_acc_values),
        "top_k": args.top_k,
        "fetch_k": args.fetch_k,
        "skip_answer": args.skip_answer,
        "index_stats": index_stats,
    }

    return {"summary": summary, "cases": per_case}


def _print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("=" * 68)
    print("RAG Evaluation Summary")
    print("=" * 68)
    print(f"Cases total          : {summary['cases_total']}")
    print(f"Source labeled cases : {summary['source_labeled_cases']}")
    print(f"Answer labeled cases : {summary['answer_labeled_cases']}")
    print(f"Recall@K (avg)       : {summary['avg_source_recall']}")
    print(f"Hit@K                : {summary['source_hit_rate']}")
    print(f"Precision-like (avg) : {summary.get('avg_source_precision_like')}")
    print(f"Top1 source hit rate : {summary.get('top1_source_hit_rate')}")
    print(f"Source MRR           : {summary.get('source_mrr')}")
    print(f"Answer accuracy (avg): {summary['avg_answer_accuracy']}")
    print(f"Top K / Fetch K      : {summary['top_k']} / {summary['fetch_k']}")
    print(f"Index stats          : {summary['index_stats']}")
    print("=" * 68)


def parse_args() -> argparse.Namespace:
    rag_dir = Path(__file__).resolve().parent
    default_cases = rag_dir / "ragtest_cases.json"
    default_out = rag_dir / "ragtest_report.json"

    parser = argparse.ArgumentParser(description="RAG recall + answer accuracy evaluator")
    parser.add_argument("--cases", default=str(default_cases), help="Path to test case JSON file")
    parser.add_argument("--out", default=str(default_out), help="Path to output report JSON file")
    parser.add_argument("--top-k", type=int, default=int(os.getenv("RETRIEVAL_TOP_K", "8")))
    parser.add_argument("--fetch-k", type=int, default=int(os.getenv("RETRIEVAL_FETCH_K", "40")))
    parser.add_argument("--skip-answer", action="store_true", help="Only evaluate retrieval")
    parser.add_argument(
        "--index-if-needed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Index new/changed local files before testing",
    )
    parser.add_argument("--reindex", action="store_true", help="Rebuild vector collection before testing")
    parser.add_argument(
        "--include-answer-text",
        action="store_true",
        help="Include full answer text in output report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(args)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_summary(report)
    print(f"Report saved to: {out_path}")


if __name__ == "__main__":
    main()
