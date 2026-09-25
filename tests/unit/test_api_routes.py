import io


def test_health_reports_zero_documents_initially(test_client):
    resp = test_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["documents_indexed"] == 0


def test_ingest_txt_file_succeeds(test_client):
    content = b"Employees receive 20 vacation days and 10 sick days per calendar year."
    files = {"file": ("handbook.txt", io.BytesIO(content), "text/plain")}
    resp = test_client.post("/ingest", files=files)
    assert resp.status_code == 201
    body = resp.json()
    assert body["filename"] == "handbook.txt"
    assert body["chunk_count"] >= 1
    assert body["status"] == "indexed"


def test_ingest_empty_file_returns_422(test_client):
    files = {"file": ("empty.txt", io.BytesIO(b"   "), "text/plain")}
    resp = test_client.post("/ingest", files=files)
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "empty_document"


def test_ingest_unsupported_extension_returns_415(test_client):
    files = {"file": ("virus.exe", io.BytesIO(b"binary"), "application/octet-stream")}
    resp = test_client.post("/ingest", files=files)
    assert resp.status_code == 415
    assert resp.json()["error_code"] == "unsupported_file_type"


def test_query_before_any_ingestion_returns_409(test_client):
    resp = test_client.post("/query", json={"question": "What is the vacation policy?"})
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "no_documents_indexed"


def test_query_with_invalid_question_returns_422(test_client):
    resp = test_client.post("/query", json={"question": "a"})
    assert resp.status_code == 422


def test_end_to_end_ingest_then_query(test_client):
    content = b"Employees receive 20 vacation days and 10 sick days per calendar year."
    files = {"file": ("handbook.txt", io.BytesIO(content), "text/plain")}
    ingest_resp = test_client.post("/ingest", files=files)
    assert ingest_resp.status_code == 201

    query_resp = test_client.post("/query", json={"question": "How many vacation days do employees get?"})
    assert query_resp.status_code == 200
    body = query_resp.json()
    assert body["citations"], "expected at least one citation from the fake LLM's canned response"
    assert body["citations"][0]["source_filename"] == "handbook.txt"


def test_re_ingesting_same_filename_replaces_old_chunks(test_client):
    v1 = {"file": ("policy.txt", io.BytesIO(b"Employees receive 15 vacation days per year." * 3), "text/plain")}
    v2 = {"file": ("policy.txt", io.BytesIO(b"Employees receive 25 vacation days per year." * 3), "text/plain")}

    r1 = test_client.post("/ingest", files=v1)
    doc_id_1 = r1.json()["document_id"]
    r2 = test_client.post("/ingest", files=v2)
    doc_id_2 = r2.json()["document_id"]

    assert doc_id_1 == doc_id_2  # same filename -> same deterministic document_id
    docs = test_client.get("/documents").json()
    assert len(docs) == 1  # not duplicated


def test_list_and_delete_documents(test_client):
    files = {"file": ("temp.txt", io.BytesIO(b"Some real content goes here for a real test."), "text/plain")}
    test_client.post("/ingest", files=files)

    docs = test_client.get("/documents").json()
    assert len(docs) == 1

    del_resp = test_client.delete("/documents/temp.txt")
    assert del_resp.status_code == 204

    docs_after = test_client.get("/documents").json()
    assert len(docs_after) == 0


def test_delete_nonexistent_document_returns_404(test_client):
    resp = test_client.delete("/documents/does_not_exist.txt")
    assert resp.status_code == 404
