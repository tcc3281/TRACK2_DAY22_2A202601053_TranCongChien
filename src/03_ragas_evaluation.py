"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import json
import warnings
from copy import deepcopy
warnings.filterwarnings("ignore")

# HACK: Bypass deprecated langchain_community VertexAI import in Ragas
import types
mock_vertex = types.ModuleType("vertexai")
mock_vertex.ChatVertexAI = object
sys.modules["langchain_community.chat_models.vertexai"] = mock_vertex

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


# ── 1. Prompt Templates (đồng bộ từ Bước 2) ──────────────────────────────────
SYSTEM_V1 = (
    "You are a helpful, polite, and concise AI assistant. "
    "Answer the user's question clearly and directly based ONLY on the provided context below (limit to 2 to 4 sentences). "
    "Do not hallucinate or extrapolate. If the context does not contain enough information, respond: "
    "'I am sorry, but I cannot find this information in the provided documentation.'\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "You are a senior data and information analyst with an expert, structured tone. "
    "Answer the question strictly according to the following guidelines (3 to 5 sentences total):\n"
    "1. Provide a direct, clear answer to the primary question.\n"
    "2. Cite specific details, facts, or definitions directly from the Context below.\n"
    "3. State the degree of completeness based on the available context.\n"
    "Base your answer strictly on the provided Context without speculation. "
    "If the information is not present, state clearly that the documentation does not provide this data.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    # Retrieve documents từ retriever
    docs = retriever.invoke(question)

    # Tạo contexts là danh sách page_content (KHÔNG ghép chuỗi ở đây)
    contexts = [doc.page_content for doc in docs]

    # Ghép contexts thành 1 string để truyền vào {context} của prompt
    ctx_str = "\n\n".join(contexts)

    # Chạy chain (prompt | llm | StrOutputParser()).invoke(...)
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context":  ctx_str,
        "question": question,
    })

    # Trả về dict với answer và contexts (list)
    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    results = []
    print(f"\n🚀 Đang chạy 50 câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        # Gọi run_rag() cho câu hỏi hiện tại
        out = run_rag(retriever, llm, prompt, qa["question"])

        # Append vào results dict với 4 keys
        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    # Tạo list các SingleTurnSample từ rag_results
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    # Wrap thành EvaluationDataset và trả về
    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: evaluate() thực hiện nhiều lần gọi LLM evaluator song song.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ ~3-7 phút)")

    # Tạo EvaluationDataset từ rag_results
    dataset = build_ragas_dataset(rag_results)

    # LLM và Embeddings riêng để RAGAS dùng làm evaluator
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig

    llm_eval = LangchainLLMWrapper(get_llm(temperature=0))
    emb_eval = LangchainEmbeddingsWrapper(get_embeddings())

    # Deep-copy metrics: metric trong RAGAS là singleton có state nội bộ (llm,
    # embeddings, cache). Dùng lại chung giữa 2 lần evaluate() có thể gây lỗi
    # hoặc trả về NaN ở lần chạy thứ 2.
    metrics = [deepcopy(m) for m in [faithfulness, answer_relevancy, context_recall, context_precision]]

    # Cấu hình RunConfig:
    # - max_workers=8: gpt-4o-mini chịu được concurrency cao. Với max_workers=2,
    #   50 samples x 4 metrics x ~40-60s/sample => chạy hàng giờ, dễ bị cancel
    #   vì timeout -> ra NaN.
    # - timeout=300: mỗi task được 5 phút, tránh bị cancel giữa chừng.
    # - max_retries=15 + max_wait=60: retry đệm exponential khi gặp rate limit.
    run_config = RunConfig(
        max_workers=8,
        timeout=300,
        max_retries=15,
        max_wait=60,
    )

    # Gọi evaluate() với đầy đủ 4 metrics và run_config
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm_eval,
        embeddings=emb_eval,
        run_config=run_config,
        show_progress=True,
    )

    # Tính mean score cho mỗi metric (lọc bỏ None và NaN an toàn)
    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw = result[key]
        valid_vals = [float(v) for v in raw if v is not None and not np.isnan(float(v))]
        scores[key] = float(np.mean(valid_vals)) if valid_vals else 0.0
        n_bad = len(raw) - len(valid_vals)
        if n_bad > 0:
            print(f"  ⚠️  {key}: {n_bad}/{len(raw)} samples bị lỗi/NaN (đã loại khỏi mean)")

    # In kết quả
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    # Tạo vectorstore
    vectorstore = setup_vectorstore()

    # Thu thập kết quả RAG cho cả V1 và V2
    # Có cache ra data/rag_outputs_{version}.json: nếu file đã tồn tại thì dùng lại,
    # không phải generate lại 50 câu trả lời khi chỉ cần chạy lại phần evaluate.
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    def get_rag_results(version: str) -> list:
        cache_path = data_dir / f"rag_outputs_{version}.json"
        if cache_path.exists():
            print(f"📂 Dùng lại cache: {cache_path}")
            return json.loads(cache_path.read_text(encoding="utf-8"))
        results = collect_rag_outputs(vectorstore, version)
        cache_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"💾 Đã cache kết quả RAG vào {cache_path}")
        return results

    v1_results = get_rag_results("v1")
    v2_results = get_rag_results("v2")

    # Chạy RAGAS evaluation
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    # Lưu báo cáo vào data/ragas_report.json
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    
    # Ghi report vào file bằng json.dumps hoặc json.dump
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"💾 Đã lưu báo cáo vào {report_path}")


if __name__ == "__main__":
    main()
