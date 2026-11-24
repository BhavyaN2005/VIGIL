import pytest


@pytest.mark.asyncio
async def test_get_topology(client):
    response = await client.get("/topology")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "summary" in data
    assert "zones" in data
    assert "nodes" in data
    assert "edges" in data
    assert "active_attack_paths" in data

    assert data["summary"]["total_nodes"] > 5
    assert data["summary"]["total_edges"] > 5

    # Check node structure
    sample_node = data["nodes"][0]
    assert "id" in sample_node
    assert "ip" in sample_node
    assert "zone" in sample_node
    assert "status" in sample_node
    assert "risk_score" in sample_node

    # Check edge structure
    sample_edge = data["edges"][0]
    assert "source" in sample_edge
    assert "target" in sample_edge
    assert "traversal_probability" in sample_edge


@pytest.mark.asyncio
async def test_isolate_node_soar_playbook(client):
    # Isolate an existing node
    node_id = "node-ws-finance"
    response = await client.post(f"/topology/node/{node_id}/isolate")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["contained_node"]["status"] == "isolated"
    assert data["contained_node"]["risk_score"] == 0.05


@pytest.mark.asyncio
async def test_isolate_nonexistent_node(client):
    response = await client.post("/topology/node/nonexistent-node-999/isolate")
    assert response.status_code == 404

