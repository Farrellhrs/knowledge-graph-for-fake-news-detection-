"""
GraphPipeline: A comprehensive knowledge graph generation pipeline for English news articles.

This module provides a complete pipeline for converting English news text into knowledge graphs
by extracting entities, normalizing them with Gemini AI, linking to Wikidata, and building
comprehensive subgraphs with all relationships.

Author: Generated from Comprehensive_Subgraph_Only.ipynb
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import json
import requests
import networkx as nx
from datetime import datetime
from collections import defaultdict
import pandas as pd
import numpy as np
import re
import subprocess

# NLP libraries
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Gemini AI
try:
    import google.generativeai as genai
except ImportError:
    print("📦 Installing google-generativeai...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai"])
    import google.generativeai as genai


class ImprovedWikidataLinker:
    """Enhanced Wikidata linker with improved error handling and multiple query methods."""
    
    def __init__(self):
        self.sparql_endpoint = "https://query.wikidata.org/sparql"
        self.base_url = "https://www.wikidata.org/w/api.php"
        self.user_agent = "KnowledgeGraphBuilder/1.0 (https://example.com/contact)"
    
    def search_wikidata_entity(self, entity_name, limit=5):
        """Search for Wikidata entities by name"""
        try:
            params = {
                'action': 'wbsearchentities',
                'search': entity_name,
                'language': 'en',
                'format': 'json',
                'limit': limit
            }
            
            headers = {'User-Agent': self.user_agent}
            response = requests.get(self.base_url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('search', [])
            return []
        except Exception as e:
            print(f"  ⚠️ Error searching for {entity_name}: {e}")
            return []
    
    def get_enhanced_relationships(self, entity_id, max_relationships=50):
        """Get enhanced relationships for an entity using SPARQL with improved error handling"""
        relationships = []
        
        # Method 1: Try comprehensive SPARQL query
        relationships.extend(self._get_sparql_relationships(entity_id, max_relationships))
        
        # Method 2: If SPARQL fails or returns few results, try API method
        if len(relationships) < 5:
            print(f"    🔄 SPARQL returned {len(relationships)} relationships, trying API fallback...")
            api_relationships = self._get_api_relationships(entity_id, max_relationships)
            relationships.extend(api_relationships)
        
        # Remove duplicates based on property
        seen_properties = set()
        unique_relationships = []
        for rel in relationships:
            prop_key = rel.get('property', '')
            if prop_key not in seen_properties:
                seen_properties.add(prop_key)
                unique_relationships.append(rel)
        
        print(f"    📊 Found {len(unique_relationships)} unique relationships")
        return unique_relationships[:max_relationships]
    
    def _get_sparql_relationships(self, entity_id, max_relationships):
        """Get relationships using SPARQL query"""
        relationships = []
        
        # Improved SPARQL query with better structure
        query = f"""
        SELECT DISTINCT ?property ?propertyLabel ?value ?valueLabel WHERE {{
          {{
            wd:{entity_id} ?property ?value .
            ?prop wikibase:directClaim ?property .
            FILTER(?property != wdt:P31)  # Exclude instance of for now
          }}
          UNION
          {{
            ?value ?property wd:{entity_id} .
            ?prop wikibase:directClaim ?property .
            FILTER(?property != wdt:P31)  # Exclude instance of for now
          }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,id" . }}
        }}
        LIMIT {max_relationships}
        """
        
        try:
            headers = {'User-Agent': self.user_agent}
            response = requests.get(
                self.sparql_endpoint,
                params={'query': query, 'format': 'json'},
                headers=headers,
                timeout=20
            )
            
            if response.status_code == 200:
                data = response.json()
                bindings = data.get('results', {}).get('bindings', [])
                print(f"    📡 SPARQL returned {len(bindings)} bindings")
                
                for binding in bindings:
                    relationships.append({
                        'property': binding.get('property', {}).get('value', ''),
                        'property_label': binding.get('propertyLabel', {}).get('value', ''),
                        'value': binding.get('value', {}).get('value', ''),
                        'value_label': binding.get('valueLabel', {}).get('value', ''),
                        'relationship_type': 'sparql'
                    })
            else:
                print(f"    ⚠️ SPARQL query failed with status {response.status_code}")
        
        except Exception as e:
            print(f"    ⚠️ SPARQL error for {entity_id}: {e}")
        
        return relationships
    
    def _get_api_relationships(self, entity_id, max_relationships):
        """Fallback method using Wikidata API to get basic entity information"""
        relationships = []
        
        try:
            # Get entity data via API
            params = {
                'action': 'wbgetentities',
                'ids': entity_id,
                'format': 'json',
                'languages': 'en|id',
                'props': 'claims'
            }
            
            headers = {'User-Agent': self.user_agent}
            response = requests.get(self.base_url, params=params, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                entity_data = data.get('entities', {}).get(entity_id, {})
                claims = entity_data.get('claims', {})
                
                print(f"    📡 API returned {len(claims)} claim types")
                
                # Process claims
                count = 0
                for prop_id, claim_list in claims.items():
                    if count >= max_relationships:
                        break
                    
                    # Get property label
                    prop_label = self._get_property_label(prop_id)
                    
                    # Process first few claims for this property
                    for claim in claim_list[:3]:  # Limit claims per property
                        if count >= max_relationships:
                            break
                        
                        # Extract main snak
                        main_snak = claim.get('mainsnak', {})
                        if main_snak.get('snaktype') == 'value':
                            datavalue = main_snak.get('datavalue', {})
                            value_type = datavalue.get('type', '')
                            
                            if value_type == 'wikibase-entityid':
                                # Entity reference
                                value_id = datavalue.get('value', {}).get('id', '')
                                if value_id:
                                    relationships.append({
                                        'property': f"http://www.wikidata.org/prop/direct/{prop_id}",
                                        'property_label': prop_label,
                                        'value': f"http://www.wikidata.org/entity/{value_id}",
                                        'value_label': value_id,  # Will be resolved later
                                        'relationship_type': 'api'
                                    })
                                    count += 1
                            
                            elif value_type in ['string', 'monolingualtext']:
                                # String value
                                value = datavalue.get('value', '')
                                if isinstance(value, dict):
                                    value = value.get('text', str(value))
                                
                                relationships.append({
                                    'property': f"http://www.wikidata.org/prop/direct/{prop_id}",
                                    'property_label': prop_label,
                                    'value': str(value),
                                    'value_label': str(value),
                                    'relationship_type': 'api'
                                })
                                count += 1
                            
                            elif value_type == 'time':
                                # Time value
                                time_value = datavalue.get('value', {}).get('time', '')
                                if time_value:
                                    relationships.append({
                                        'property': f"http://www.wikidata.org/prop/direct/{prop_id}",
                                        'property_label': prop_label,
                                        'value': time_value,
                                        'value_label': time_value,
                                        'relationship_type': 'api'
                                    })
                                    count += 1
            
            else:
                print(f"    ⚠️ API request failed with status {response.status_code}")
        
        except Exception as e:
            print(f"    ⚠️ API error for {entity_id}: {e}")
        
        return relationships
    
    def _get_property_label(self, prop_id):
        """Get human-readable label for a property"""
        try:
            params = {
                'action': 'wbgetentities',
                'ids': prop_id,
                'format': 'json',
                'languages': 'en',
                'props': 'labels'
            }
            
            headers = {'User-Agent': self.user_agent}
            response = requests.get(self.base_url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                labels = data.get('entities', {}).get(prop_id, {}).get('labels', {})
                return labels.get('en', {}).get('value', prop_id)
        
        except Exception:
            pass
        
        return prop_id


class GeminiEntityNormalizer:
    """Entity normalizer using Gemini AI for better standardization."""
    
    def __init__(self, model):
        self.model = model
    
    def normalize_entities(self, entities_dict):
        """Normalize entities using Gemini AI for better standardization"""
        print("🧠 Normalizing entities using Gemini AI...")
        
        # Prepare entities list for batch processing
        entity_names = list(entities_dict.keys())
        
        # Create prompt for entity normalization
        prompt = f"""
        You are an expert entity normalizer for English news articles. Your task is to normalize and standardize the following extracted entities to improve their accuracy for knowledge graph construction.

        Instructions:
        1. Clean up any OCR errors or tokenization artifacts
        2. Expand abbreviations to full forms when appropriate
        3. Standardize proper nouns to their most common form
        4. For organizations, use official names
        5. For locations, use standard geographical names
        6. For persons, use proper name formatting
        7. Use proper English capitalization and spelling

        Original entities to normalize:
        {', '.join(entity_names)}

        Please return the normalized entities in the following JSON format:
        {{
            "original_entity_1": "normalized_entity_1",
            "original_entity_2": "normalized_entity_2",
            ...
        }}

        Only return the JSON object, no additional text.
        """
        
        try:
            print("  🤖 Sending normalization request to Gemini...")
            response = self.model.generate_content(prompt)
            
            # Parse the JSON response
            response_text = response.text.strip()
            
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                normalization_mapping = json.loads(json_str)
                
                print(f"  ✅ Successfully normalized {len(normalization_mapping)} entities")
                
                # Apply normalization to entities
                normalized_entities = {}
                for original_name, entity_data in entities_dict.items():
                    normalized_name = normalization_mapping.get(original_name, original_name)
                    
                    # Update entity data with normalized name
                    normalized_entities[normalized_name] = {
                        **entity_data,
                        'original_extracted_name': original_name,
                        'normalized_by_gemini': normalized_name != original_name
                    }
                    
                    if normalized_name != original_name:
                        print(f"  📝 '{original_name}' → '{normalized_name}'")
                
                return normalized_entities, normalization_mapping
            
            else:
                print("  ⚠️ Could not parse JSON from Gemini response, using original entities")
                return entities_dict, {}
                
        except Exception as e:
            print(f"  ❌ Error during normalization: {e}")
            print("  ℹ️ Proceeding with original entities")
            return entities_dict, {}


class ImprovedComprehensiveKnowledgeGraphBuilder:
    """Comprehensive knowledge graph builder with improved debugging and error handling."""
    
    def __init__(self, wikidata_linker):
        self.wikidata_linker = wikidata_linker
    
    def build_entity_subgraph(self, entity_id, entity_data, max_depth=2, max_relationships=30):
        """Build comprehensive subgraph for a single entity with better debugging"""
        subgraph = nx.Graph()
        processed = set()
        to_process = [(entity_id, 0)]  # (entity_id, depth)
        
        print(f"  🔧 Building subgraph for {entity_id} (max_depth={max_depth}, max_rel={max_relationships})")
        
        while to_process:
            current_entity, depth = to_process.pop(0)
            
            if current_entity in processed or depth > max_depth:
                continue
            
            processed.add(current_entity)
            print(f"    📍 Processing {current_entity} at depth {depth}")
            
            # Add entity node
            if current_entity == entity_id:
                # Main entity
                subgraph.add_node(current_entity, 
                                label=entity_data['wikidata_label'],
                                type='main_entity',
                                original_names=entity_data.get('original_names', [entity_data.get('original_name', current_entity)]),
                                normalized_name=entity_data.get('normalized_name', current_entity),
                                description=entity_data['description'],
                                wikidata_id=current_entity,
                                was_normalized=entity_data.get('was_normalized', False),
                                was_deduplicated=entity_data.get('was_deduplicated', False),
                                entity_type=entity_data.get('entity_type', 'unknown'),
                                confidence=entity_data.get('confidence', 0.0))
            else:
                # Related entity
                subgraph.add_node(current_entity,
                                label=current_entity,
                                type='related_entity',
                                wikidata_id=current_entity)
            
            # Get relationships with retry logic
            print(f"    🔍 Getting relationships for {current_entity}...")
            max_rel_for_depth = max_relationships // (2 ** depth) if depth > 0 else max_relationships
            relationships = self.wikidata_linker.get_enhanced_relationships(
                current_entity, max_relationships=max_rel_for_depth
            )
            
            if not relationships:
                print(f"    ⚠️ No relationships found for {current_entity}")
                # Add a basic node info if no relationships found
                if depth == 0:  # Only for main entities
                    # Try to get basic instance-of relationship
                    basic_rels = self._get_basic_relationships(current_entity)
                    relationships.extend(basic_rels)
            
            edges_added = 0
            # Process each relationship
            for i, rel in enumerate(relationships):
                if not rel.get('property') or not rel.get('value'):
                    continue
                    
                try:
                    prop_id = rel['property'].split('/')[-1] if 'property' in rel else f'unknown_{i}'
                    prop_label = rel.get('property_label', prop_id)
                    
                    # Create relationship node
                    rel_node_id = f"REL_{current_entity}_{prop_id}_{i}"
                    subgraph.add_node(rel_node_id,
                                    label=prop_label,
                                    type='relationship',
                                    property_id=prop_id,
                                    source_entity=current_entity)
                    
                    # Connect entity to relationship
                    subgraph.add_edge(current_entity, rel_node_id,
                                    type='has_property',
                                    weight=1.0)
                    edges_added += 1
                    
                    # Handle value
                    if 'value' in rel and rel['value']:
                        value = rel['value']
                        value_label = rel.get('value_label', str(value))
                        
                        if isinstance(value, str) and ('wikidata.org/entity/Q' in value or value.startswith('Q')):
                            # Wikidata entity
                            if 'wikidata.org/entity/' in value:
                                value_id = value.split('/')[-1]
                            else:
                                value_id = value
                            
                            if not subgraph.has_node(value_id):
                                subgraph.add_node(value_id,
                                                label=value_label if value_label != value_id else value_id,
                                                type='linked_entity',
                                                wikidata_id=value_id)
                            
                            subgraph.add_edge(rel_node_id, value_id,
                                            type='has_value',
                                            weight=1.0)
                            edges_added += 1
                            
                            # Add to processing queue for deeper exploration
                            if depth < max_depth and len(to_process) < 20 and value_id not in processed:
                                to_process.append((value_id, depth + 1))
                        
                        else:
                            # Literal value
                            value_node_id = f"LIT_{current_entity}_{hash(str(value)) % 100000}"
                            if not subgraph.has_node(value_node_id):
                                subgraph.add_node(value_node_id,
                                                label=str(value_label)[:100] if value_label else str(value)[:100],
                                                type='literal',
                                                value=str(value)[:200])
                            
                            subgraph.add_edge(rel_node_id, value_node_id,
                                            type='has_literal_value',
                                            weight=1.0)
                            edges_added += 1
                
                except Exception as e:
                    print(f"    ⚠️ Error processing relationship {i} for {current_entity}: {e}")
                    continue
            
            print(f"    ✅ Added {edges_added} edges for {current_entity}")
        
        final_nodes = subgraph.number_of_nodes()
        final_edges = subgraph.number_of_edges()
        print(f"  🎯 Subgraph complete: {final_nodes} nodes, {final_edges} edges")
        
        return subgraph
    
    def _get_basic_relationships(self, entity_id):
        """Get basic relationships as fallback"""
        basic_relationships = []
        
        # Try to get instance-of (P31) relationship specifically
        query = f"""
        SELECT ?value ?valueLabel WHERE {{
          wd:{entity_id} wdt:P31 ?value .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
        }}
        LIMIT 5
        """
        
        try:
            headers = {'User-Agent': self.wikidata_linker.user_agent}
            response = requests.get(
                self.wikidata_linker.sparql_endpoint,
                params={'query': query, 'format': 'json'},
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                for binding in data.get('results', {}).get('bindings', []):
                    basic_relationships.append({
                        'property': 'http://www.wikidata.org/prop/direct/P31',
                        'property_label': 'instance of',
                        'value': binding.get('value', {}).get('value', ''),
                        'value_label': binding.get('valueLabel', {}).get('value', ''),
                        'relationship_type': 'basic_fallback'
                    })
        
        except Exception as e:
            print(f"    ⚠️ Basic relationship query failed: {e}")
        
        return basic_relationships
    
    def build_comprehensive_graph(self, linked_entities):
        """Build comprehensive knowledge graph from all entities with improved debugging"""
        print("🏗️ Building comprehensive knowledge graph with improved debugging...")
        
        comprehensive_graph = nx.Graph()
        
        for i, (entity_id, entity_data) in enumerate(linked_entities.items()):
            print(f"\n🔍 Processing entity {i+1}/{len(linked_entities)}: {entity_id}")
            print(f"  📝 Label: {entity_data['wikidata_label']}")
            
            # Build subgraph for this entity
            try:
                entity_subgraph = self.build_entity_subgraph(entity_id, entity_data)
                
                print(f"  📊 Subgraph built: {entity_subgraph.number_of_nodes()} nodes, {entity_subgraph.number_of_edges()} edges")
                
                # Merge into comprehensive graph
                before_nodes = comprehensive_graph.number_of_nodes()
                before_edges = comprehensive_graph.number_of_edges()
                
                comprehensive_graph = nx.compose(comprehensive_graph, entity_subgraph)
                
                after_nodes = comprehensive_graph.number_of_nodes()
                after_edges = comprehensive_graph.number_of_edges()
                
                print(f"  📈 Graph growth: +{after_nodes - before_nodes} nodes, +{after_edges - before_edges} edges")
                
            except Exception as e:
                print(f"  ❌ Error building subgraph for {entity_id}: {e}")
                # Add at least the main entity node
                comprehensive_graph.add_node(entity_id, 
                                           label=entity_data['wikidata_label'],
                                           type='main_entity_error',
                                           error=str(e))
        
        # Create a central interaction node to connect all main entities
        main_entity_nodes = []
        
        # Identify all nodes with type 'main_entity' or 'main_entity_error'
        for node, attrs in comprehensive_graph.nodes(data=True):
            node_type = attrs.get('type', '')
            if node_type == 'main_entity' or node_type == 'main_entity_error':
                main_entity_nodes.append(node)
        
        # Add a central interaction node
        interaction_node_id = 'interaction_node'
        comprehensive_graph.add_node(interaction_node_id,
                                   label='Interaction Hub',
                                   type='interaction',
                                   source='news_context')
        
        # Connect each main entity to the interaction node
        contextual_edges = 0
        for entity_id in main_entity_nodes:
            comprehensive_graph.add_edge(entity_id, interaction_node_id,
                                       type='entity_interaction',
                                       weight=1.0,
                                       source='news_context')
            contextual_edges += 1
        
        print(f"\n🔗 Added central interaction node with {contextual_edges} connections to main entities")
        print(f"\n🎯 Comprehensive graph completed!")
        print(f"  📊 Total nodes: {comprehensive_graph.number_of_nodes()}")
        print(f"  🔗 Total edges: {comprehensive_graph.number_of_edges()}")
        
        return comprehensive_graph


class GraphPipeline:
    """
    A comprehensive knowledge graph generation pipeline for English news articles.
    
    This class encapsulates the complete process of:
    1. Extracting entities using English BERT NER
    2. Normalizing entities using Gemini AI
    3. Linking entities to Wikidata
    4. Building comprehensive knowledge graphs with all relationships
    """
    
    def __init__(self, gemini_api_key, model_name="dslim/bert-base-NER"):
        """
        Initialize the GraphPipeline with all necessary models and components.
        
        Args:
            gemini_api_key (str): API key for Gemini AI
            model_name (str): Name of the English BERT NER model
        """
        print("🚀 Initializing GraphPipeline...")
        
        # Configure Gemini API
        self.gemini_api_key = gemini_api_key
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-lite')
        print("✅ Gemini API configured")
        
        # Load English BERT NER model
        print("🤖 Loading English BERT NER model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner_pipeline = pipeline(
            "ner", 
            model=self.ner_model, 
            tokenizer=self.tokenizer, 
            aggregation_strategy="simple"
        )
        print("✅ English BERT NER model loaded")
        
        # Initialize components
        self.wikidata_linker = ImprovedWikidataLinker()
        self.entity_normalizer = GeminiEntityNormalizer(self.gemini_model)
        self.graph_builder = ImprovedComprehensiveKnowledgeGraphBuilder(self.wikidata_linker)
        
        print("✅ GraphPipeline initialization completed!")
    
    def extract_entities(self, news_text):
        """
        Extract entities from English news text using BERT NER.
        
        Args:
            news_text (str): The news text to process
            
        Returns:
            dict: Dictionary of processed entities with metadata
        """
        print("🔍 Extracting entities from news text...")
        
        # Simple character-based truncation to prevent tensor size mismatch
        # Conservative limit: ~1500 chars should be well under 512 tokens for English
        MAX_CHARS = 1500
        
        if len(news_text) > MAX_CHARS:
            truncated_text = news_text[:MAX_CHARS]
            print(f"  📏 Text truncated from {len(news_text)} to {len(truncated_text)} characters")
        else:
            truncated_text = news_text
        
        entities = self.ner_pipeline(truncated_text)
        
        print(f"Found {len(entities)} entities:")
        for i, entity in enumerate(entities, 1):
            print(f"{i}. '{entity['word']}' [{entity['entity_group']}] (confidence: {entity['score']:.3f})")
        
        # Process entities for knowledge graph
        processed_entities = {}
        for entity in entities:
            clean_word = entity['word'].replace('##', '').strip()
            if len(clean_word) > 2 and entity['score'] > 0.5:  # Filter low confidence
                processed_entities[clean_word] = {
                    'type': entity['entity_group'],
                    'confidence': entity['score'],
                    'position': (entity['start'], entity['end'])
                }
        
        print(f"✅ Processed {len(processed_entities)} high-confidence entities")
        return processed_entities
    
    def normalize_and_deduplicate_entities(self, processed_entities):
        """
        Normalize entities using Gemini AI and deduplicate them.
        
        Args:
            processed_entities (dict): Dictionary of processed entities
            
        Returns:
            tuple: (deduplicated_entities, original_names_mapping)
        """
        print("🧠 Step 1: Entity Normalization with Gemini AI")
        print("=" * 60)
        
        # Normalize entities
        normalized_entities, normalization_mapping = self.entity_normalizer.normalize_entities(processed_entities)
        
        # Display normalization summary
        total_entities = len(processed_entities)
        normalized_count = sum(1 for orig, norm in normalization_mapping.items() if orig != norm)
        
        print(f"\n📊 Normalization Summary:")
        print(f"  - Total entities: {total_entities}")
        print(f"  - Normalized: {normalized_count}")
        print(f"  - Unchanged: {total_entities - normalized_count}")
        print(f"  - Normalization rate: {(normalized_count / total_entities * 100) if total_entities > 0 else 0:.1f}%")
        
        # Deduplicate normalized entities
        print(f"\n🔧 Step 2: Deduplicating normalized entities...")
        print("=" * 60)
        
        deduplicated_entities = {}
        original_names_mapping = {}
        
        for normalized_name, entity_data in normalized_entities.items():
            if normalized_name not in deduplicated_entities:
                # First occurrence of this normalized name
                deduplicated_entities[normalized_name] = entity_data.copy()
                original_names_mapping[normalized_name] = [entity_data.get('original_extracted_name', normalized_name)]
            else:
                # Duplicate normalized name - merge the information
                original_names_mapping[normalized_name].append(entity_data.get('original_extracted_name', normalized_name))
                
                # Keep the entity with higher confidence
                if entity_data['confidence'] > deduplicated_entities[normalized_name]['confidence']:
                    deduplicated_entities[normalized_name] = entity_data.copy()
        
        print(f"📊 Deduplication Results:")
        print(f"  - Before deduplication: {len(normalized_entities)} entities")
        print(f"  - After deduplication: {len(deduplicated_entities)} unique entities")
        print(f"  - Duplicates removed: {len(normalized_entities) - len(deduplicated_entities)}")
        
        return deduplicated_entities, original_names_mapping
    
    def link_entities_to_wikidata(self, deduplicated_entities, original_names_mapping):
        """
        Link deduplicated entities to Wikidata.
        
        Args:
            deduplicated_entities (dict): Dictionary of deduplicated entities
            original_names_mapping (dict): Mapping of original names for each entity
            
        Returns:
            dict: Dictionary of linked entities with Wikidata information
        """
        print("🔗 Linking deduplicated entities to Wikidata...")
        print("=" * 60)
        
        linked_entities = {}
        failed_links = []
        
        for entity_name, entity_data in deduplicated_entities.items():
            print(f"\n🔍 Searching for: '{entity_name}'")
            
            # Show consolidation info
            original_names = original_names_mapping[entity_name]
            if len(original_names) > 1:
                print(f"  🔗 Consolidated from: {original_names}")
            elif entity_data.get('normalized_by_gemini', False):
                print(f"  📝 Normalized from: '{entity_data['original_extracted_name']}'")
            
            try:
                # Search for Wikidata entities
                search_results = self.wikidata_linker.search_wikidata_entity(entity_name, limit=3)
                
                if search_results and len(search_results) > 0:
                    # Take the best match (first result)
                    best_match = search_results[0]
                    entity_id = best_match['id']
                    entity_label = best_match.get('label', entity_name)
                    
                    # Ensure we don't overwrite existing entities with the same Wikidata ID
                    if entity_id not in linked_entities:
                        linked_entities[entity_id] = {
                            'original_names': original_names.copy(),
                            'normalized_name': entity_name,
                            'wikidata_label': entity_label,
                            'description': best_match.get('description', ''),
                            'entity_type': entity_data['type'],
                            'confidence': entity_data['confidence'],
                            'was_normalized': entity_data.get('normalized_by_gemini', False),
                            'was_deduplicated': len(original_names) > 1
                        }
                        
                        print(f"  ✅ Linked to: {entity_id} ({entity_label})")
                        
                        # Show other potential matches for verification
                        if len(search_results) > 1:
                            other_matches = [f"{r['id']} ({r.get('label', 'N/A')})" for r in search_results[1:3]]
                            print(f"     Other matches: {other_matches}")
                    else:
                        print(f"  ⚠️ Wikidata ID {entity_id} already used, skipping duplicate")
                        failed_links.append((entity_name, f"Duplicate Wikidata ID: {entity_id}"))
                else:
                    print(f"  ❌ No Wikidata match found")
                    failed_links.append((entity_name, "No matches found"))
                    
            except Exception as e:
                print(f"  ⚠️ Error linking '{entity_name}': {e}")
                failed_links.append((entity_name, str(e)))
        
        print(f"\n✅ Successfully linked {len(linked_entities)} unique entities to Wikidata")
        
        if failed_links:
            print(f"\n⚠️ Failed to link {len(failed_links)} entities:")
            for entity_name, reason in failed_links:
                print(f"  - '{entity_name}': {reason}")
        
        success_rate = (len(linked_entities)/len(deduplicated_entities)*100) if deduplicated_entities else 0
        print(f"\n🎯 Linking Results:")
        print(f"  - Successfully linked: {len(linked_entities)}")
        print(f"  - Success rate: {success_rate:.1f}%")
        
        return linked_entities
    
    def generate_graph(self, news_text):
        """
        Generate a comprehensive knowledge graph from English news text.
        
        This method performs the complete pipeline:
        1. Extract entities using BERT NER
        2. Normalize entities using Gemini AI
        3. Deduplicate entities
        4. Link entities to Wikidata
        5. Build comprehensive knowledge graph
        
        Args:
            news_text (str): The English news text to process
            
        Returns:
            networkx.Graph: A comprehensive knowledge graph
        """
        print("🌐 Starting comprehensive knowledge graph generation...")
        print("=" * 70)
        
        # Step 1: Extract entities
        processed_entities = self.extract_entities(news_text)
        
        if not processed_entities:
            print("⚠️ No entities found in the text")
            return nx.Graph()
        
        # Step 2: Normalize and deduplicate entities
        deduplicated_entities, original_names_mapping = self.normalize_and_deduplicate_entities(processed_entities)
        
        if not deduplicated_entities:
            print("⚠️ No entities after normalization and deduplication")
            return nx.Graph()
        
        # Step 3: Link entities to Wikidata
        linked_entities = self.link_entities_to_wikidata(deduplicated_entities, original_names_mapping)
        
        if not linked_entities:
            print("⚠️ No entities successfully linked to Wikidata")
            return nx.Graph()
        
        # Step 4: Build comprehensive knowledge graph
        print("\n🏗️ Building comprehensive knowledge graph...")
        print("=" * 70)
        
        comprehensive_graph = self.graph_builder.build_comprehensive_graph(linked_entities)
        
        # Final summary
        print("\n🎯 KNOWLEDGE GRAPH GENERATION COMPLETED!")
        print("=" * 70)
        print(f"📊 Final graph statistics:")
        print(f"  - Total nodes: {comprehensive_graph.number_of_nodes()}")
        print(f"  - Total edges: {comprehensive_graph.number_of_edges()}")
        print(f"  - Main entities: {len(linked_entities)}")
        
        # Check connectivity of main entities
        main_entities_with_edges = 0
        for entity_id in linked_entities.keys():
            if comprehensive_graph.has_node(entity_id):
                degree = comprehensive_graph.degree(entity_id)
                if degree > 0:
                    main_entities_with_edges += 1
        
        print(f"  - Main entities with connections: {main_entities_with_edges}/{len(linked_entities)}")
        
        return comprehensive_graph


# Example usage and testing
if __name__ == "__main__":
    # Example English news text
    sample_news = """Washington, June 18, 2025 — The U.S. government officially launched an electric vehicle subsidy program with incentives of up to $7,000 per unit starting this month. This initiative is part of efforts to reduce carbon emissions and promote the use of environmentally friendly vehicles.

Secretary of Transportation, Pete Buttigieg, stated that the program targets the distribution of 200,000 subsidized electric vehicles by the end of 2025. Subsidies will be provided to consumers who meet certain criteria, including small business owners, social assistance recipients, and low-income households.

"This is a concrete manifestation of the government's commitment to accelerating the clean energy transition," said Buttigieg during a press conference in Washington.

The government has also partnered with several local electric vehicle manufacturers to ensure the availability and quality of vehicles to be distributed."""
    
    print("🧪 GraphPipeline Example Usage")
    print("=" * 50)
    
    # Note: Replace with your actual Gemini API key
    GEMINI_API_KEY = "AIzaSyAvMQynyVcZqvSqPQF3JRZDVaco5Lpa7iE"
    
    try:
        # Initialize pipeline
        pipeline = GraphPipeline(gemini_api_key=GEMINI_API_KEY)
        
        # Generate knowledge graph
        knowledge_graph = pipeline.generate_graph(sample_news)
        
        print(f"\n✅ Knowledge graph generated successfully!")
        print(f"Graph has {knowledge_graph.number_of_nodes()} nodes and {knowledge_graph.number_of_edges()} edges")
        
    except Exception as e:
        print(f"❌ Error in example usage: {e}")
        print("Please ensure you have a valid Gemini API key and internet connection.")
