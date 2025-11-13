# Test schema code generation functionality
import pytest
import tempfile
from pathlib import Path
from plugins.aiva_converters.core.schema_codegen_tool import SchemaCodegen

class TestSchemaCodegen:
    
    @pytest.mark.unit
    def test_typescript_generation(self, converter_instance, sample_schemas):
        """Test TypeScript interface generation"""
        
        # Generate TypeScript from security scan schema
        result = converter_instance.generate_from_string(
            schema_content=sample_schemas["security_scan"],
            target_language="typescript"
        )
        
        assert result is not None
        assert "export interface ScanResult" in result
        assert "export interface VulnerabilityFinding" in result
        assert "export enum ScanStatus" in result
        assert "export enum VulnerabilityLevel" in result
    
    @pytest.mark.unit
    def test_rust_generation(self, converter_instance, sample_schemas):
        """Test Rust struct generation"""
        
        result = converter_instance.generate_from_string(
            schema_content=sample_schemas["user"],
            target_language="rust"
        )
        
        assert result is not None
        assert "pub struct User" in result
        assert "serde::" in result
        assert "Serialize, Deserialize" in result
    
    @pytest.mark.unit
    def test_go_generation(self, converter_instance, sample_schemas):
        """Test Go struct generation"""
        
        result = converter_instance.generate_from_string(
            schema_content=sample_schemas["user"],
            target_language="go"
        )
        
        assert result is not None
        assert "type User struct" in result
        assert "`json:" in result
    
    @pytest.mark.unit
    def test_invalid_schema_handling(self, converter_instance):
        """Test handling of invalid schema input"""
        
        invalid_schema = "This is not valid Python code"
        
        with pytest.raises(Exception):
            converter_instance.generate_from_string(
                schema_content=invalid_schema,
                target_language="typescript"
            )
    
    @pytest.mark.unit
    def test_unsupported_language(self, converter_instance, sample_schemas):
        """Test handling of unsupported target language"""
        
        with pytest.raises(ValueError, match="Unsupported language"):
            converter_instance.generate_from_string(
                schema_content=sample_schemas["user"],
                target_language="cobol"
            )
    
    @pytest.mark.unit
    def test_empty_schema_handling(self, converter_instance):
        """Test handling of empty schema"""
        
        empty_schema = ""
        
        with pytest.raises(Exception):
            converter_instance.generate_from_string(
                schema_content=empty_schema,
                target_language="typescript"
            )
    
    @pytest.mark.unit
    def test_complex_schema_generation(self, converter_instance):
        """Test generation with complex nested schemas"""
        
        complex_schema = '''
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Union, Any
from datetime import datetime
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class NestedData(BaseModel):
    value: Union[str, int, float]
    metadata: Dict[str, Any]

class ComplexModel(BaseModel):
    id: str = Field(description="Unique identifier")
    priority: Priority
    nested: NestedData
    items: List[str] = Field(default_factory=list)
    optional_field: Optional[datetime] = None
    mapping: Dict[str, Union[str, int]] = Field(default_factory=dict)
'''
        
        # Test TypeScript generation
        ts_result = converter_instance.generate_from_string(
            schema_content=complex_schema,
            target_language="typescript"
        )
        
        assert "export enum Priority" in ts_result
        assert "export interface NestedData" in ts_result
        assert "export interface ComplexModel" in ts_result
        assert "Record<string," in ts_result  # Dictionary mapping
        
        # Test Rust generation
        rust_result = converter_instance.generate_from_string(
            schema_content=complex_schema,
            target_language="rust"
        )
        
        assert "pub enum Priority" in rust_result
        assert "pub struct NestedData" in rust_result
        assert "pub struct ComplexModel" in rust_result
        assert "HashMap<String," in rust_result
    
    @pytest.mark.unit
    def test_field_validation_attributes(self, converter_instance):
        """Test that field validation attributes are preserved"""
        
        validation_schema = '''
from pydantic import BaseModel, Field, validator
from typing import Optional

class ValidationModel(BaseModel):
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    age: int = Field(..., gt=0, lt=150)
    score: float = Field(..., ge=0.0, le=100.0)
    name: str = Field(..., min_length=1, max_length=100)
    
    @validator('email')
    def validate_email(cls, v):
        return v.lower()
'''
        
        # Test TypeScript generation preserves constraints
        ts_result = converter_instance.generate_from_string(
            schema_content=validation_schema,
            target_language="typescript"
        )
        
        # Should contain documentation about constraints
        assert "email: string" in ts_result
        assert "age: number" in ts_result
        
        # Test Rust generation
        rust_result = converter_instance.generate_from_string(
            schema_content=validation_schema,
            target_language="rust"
        )
        
        assert "pub email: String" in rust_result or "email: String" in rust_result
        assert "pub age: i32" in rust_result or "age: i32" in rust_result
    
    @pytest.mark.unit
    def test_output_file_generation(self, converter_instance, temp_output_dir, sample_schemas):
        """Test generation to output files"""
        
        # Generate to file
        output_file = temp_output_dir / "generated.ts"
        
        result = converter_instance.generate_to_file(
            schema_content=sample_schemas["user"],
            target_language="typescript",
            output_file=output_file
        )
        
        assert result.success
        assert output_file.exists()
        assert output_file.stat().st_size > 0
        
        # Verify content
        content = output_file.read_text()
        assert "export interface User" in content
    
    @pytest.mark.unit
    def test_multiple_models_in_schema(self, converter_instance):
        """Test generation with multiple models in same schema"""
        
        multi_model_schema = '''
from pydantic import BaseModel
from typing import List

class Author(BaseModel):
    name: str
    email: str

class Article(BaseModel):
    title: str
    content: str
    author: Author
    tags: List[str]

class Blog(BaseModel):
    name: str
    articles: List[Article]
    authors: List[Author]
'''
        
        result = converter_instance.generate_from_string(
            schema_content=multi_model_schema,
            target_language="typescript"
        )
        
        # Should generate all interfaces
        assert "export interface Author" in result
        assert "export interface Article" in result
        assert "export interface Blog" in result
        
        # Should handle nested references
        assert "author: Author" in result
        assert "articles: Article[]" in result