"""Simple tests that don't require complex dependencies"""

def test_basic_functionality():
    """Test basic Python functionality"""
    assert 1 + 1 == 2

def test_imports():
    """Test that we can import basic modules"""
    import json
    import os
    import sys
    assert json is not None
    assert os is not None
    assert sys is not None

def test_string_operations():
    """Test string operations work"""
    test_string = "AI Customer Query System"
    assert len(test_string) > 0
    assert "AI" in test_string
    assert test_string.lower() == "ai customer query system"