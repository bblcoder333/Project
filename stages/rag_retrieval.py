def retrieve_knowledge(ui_data: dict, kb, top_k: int = 5) -> list[dict]:
    description = ui_data.get("description", "")
    topic       = ui_data.get("topic", "unknown")
    elements    = ui_data.get("structured_elements", "")

    query = (
        f"Topic: {topic}. "
        f"Screen summary: {description[:200]}. "
        f"Key components: {elements[:200]}"
    )

    results = kb.query(
        query_texts=[query],
        n_results=min(top_k, kb.count()),
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({
            "text":      doc,
            "type":      meta.get("type", "unknown"),
            "relevance": round(1 - dist, 3),
        })
        print(f"  [{meta.get('type','?'):12s}] (relevance: {1-dist:.2f}) {doc[:80]}...")

    return retrieved