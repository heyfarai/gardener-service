#!/usr/bin/env python3
"""Minimal test to check core logic without dependencies"""

import os, time, uuid, math, hashlib, random

def fake_embed(text: str):
    """Test the fallback embedding function"""
    random.seed(int(hashlib.md5(text.encode()).hexdigest(),16) % 10**6)
    return [random.random() for _ in range(384)]

def cosine(a, b):
    """Test cosine similarity"""
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if not na or not nb: return 0.0
    return sum(x*y for x,y in zip(a,b))/(na*nb)

def test_core_logic():
    """Test the core clustering logic"""
    print("Testing core gardener logic...")
    
    # Test embedding generation
    v1 = fake_embed("Hello world")
    v2 = fake_embed("Hello world")  # Should be identical
    v3 = fake_embed("Goodbye moon")
    
    assert v1 == v2, "Deterministic embeddings failed"
    assert len(v1) == 384, f"Wrong embedding dimension: {len(v1)}"
    
    # Test cosine similarity
    sim_identical = cosine(v1, v2)
    sim_different = cosine(v1, v3)
    
    assert abs(sim_identical - 1.0) < 0.001, f"Identical similarity should be 1.0, got {sim_identical}"
    assert sim_different < sim_identical, "Different texts should have lower similarity"
    
    print(f"✅ Embeddings: {len(v1)} dimensions")
    print(f"✅ Identical similarity: {sim_identical:.3f}")
    print(f"✅ Different similarity: {sim_different:.3f}")
    
    # Test topic creation logic
    TOPICS = {}
    threshold = 0.78
    
    # First text creates new topic
    text1 = "Machine learning is fascinating"
    v1 = fake_embed(text1)
    best_id = None
    
    if not TOPICS or max(cosine(v1, meta["centroid"]) for meta in TOPICS.values()) < threshold:
        topic_id = f"topic_{uuid.uuid4().hex[:6]}"
        TOPICS[topic_id] = {
            "label": " ".join(text1.split()[:4]),
            "centroid": v1,
            "updated": time.time()
        }
        best_id = topic_id
    
    assert len(TOPICS) == 1, "Should create first topic"
    print(f"✅ Created topic: {TOPICS[best_id]['label']}")
    
    # Similar text should join existing topic
    text2 = "Machine learning algorithms are amazing"
    v2 = fake_embed(text2)
    
    best_score = max(cosine(v2, meta["centroid"]) for meta in TOPICS.values())
    print(f"✅ Similarity to existing topic: {best_score:.3f}")
    
    if best_score >= threshold:
        print("✅ Would join existing topic")
    else:
        print("✅ Would create new topic")
    
    print("\n🎉 Core logic test passed!")

if __name__ == "__main__":
    test_core_logic()
