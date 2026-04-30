"""
Graph I/O Utilities

Provides compatible functions for saving and loading NetworkX graphs
across different NetworkX versions.
"""

import pickle
import networkx as nx
import os

def save_graph(graph, filepath):
    """
    Save a NetworkX graph to file using pickle for maximum compatibility.
    
    Args:
        graph (nx.Graph): NetworkX graph to save
        filepath (str): Path to save the graph to
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure directory exists (only if filepath contains a directory)
        directory = os.path.dirname(filepath)
        if directory:  # Only create directory if it's not empty
            os.makedirs(directory, exist_ok=True)
        
        # Save using pickle for compatibility
        with open(filepath, 'wb') as f:
            pickle.dump(graph, f)
        return True
    except Exception as e:
        print(f"Error saving graph to {filepath}: {e}")
        return False

def load_graph(filepath):
    """
    Load a NetworkX graph from file with improved compatibility.
    
    Args:
        filepath (str): Path to the graph file
    
    Returns:
        nx.Graph or None: Loaded graph or None if failed
    """
    try:
        # Method 1: Try pickle first (our standard format)
        with open(filepath, 'rb') as f:
            graph = pickle.load(f)
        return graph
    except Exception as e:
        # Method 2: Try NetworkX gpickle methods
        try:
            # Try different NetworkX gpickle methods based on version
            if hasattr(nx, 'read_gpickle'):
                # Older NetworkX versions
                return nx.read_gpickle(filepath)
            elif hasattr(nx.readwrite, 'gpickle') and hasattr(nx.readwrite.gpickle, 'read_gpickle'):
                # Newer NetworkX versions
                return nx.readwrite.gpickle.read_gpickle(filepath)
            else:
                # Try direct import
                from networkx.readwrite import gpickle
                return gpickle.read_gpickle(filepath)
        except Exception as e2:
            print(f"Error loading graph from {filepath}: NetworkX gpickle error: {e2}")
            
            # Method 3: Try reading as generic pickle
            try:
                import pickle
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    if hasattr(data, 'nodes') and hasattr(data, 'edges'):
                        return data
                    else:
                        print(f"File {filepath} does not contain a valid NetworkX graph")
                        return None
            except Exception as e3:
                print(f"Final error loading graph from {filepath}: {e3}")
                return None

def get_graph_info(graph):
    """
    Get basic information about a graph.
    
    Args:
        graph (nx.Graph): NetworkX graph
    
    Returns:
        dict: Graph information
    """
    if graph is None:
        return {'nodes': 0, 'edges': 0, 'error': 'Graph is None'}
    
    try:
        info = {
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'is_directed': graph.is_directed(),
        }
        
        # Get node types if available
        node_types = {}
        for node, data in graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        info['node_types'] = node_types
        
        # Get edge types if available
        edge_types = {}
        for source, target, data in graph.edges(data=True):
            edge_type = data.get('type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        info['edge_types'] = edge_types
        
        return info
    except Exception as e:
        return {'nodes': 0, 'edges': 0, 'error': str(e)}

def verify_graph_file(filepath):
    """
    Verify that a graph file can be loaded successfully.
    
    Args:
        filepath (str): Path to the graph file
    
    Returns:
        bool: True if file can be loaded, False otherwise
    """
    graph = load_graph(filepath)
    return graph is not None

# Test the utilities
if __name__ == "__main__":
    # Create a test graph
    test_graph = nx.Graph()
    test_graph.add_node(1, label="Test Node", type="test")
    test_graph.add_node(2, label="Another Node", type="test")
    test_graph.add_edge(1, 2, type="test_edge", weight=1.0)
    
    # Test save/load
    test_file = "test_graph_compatibility.gpickle"
    
    print("Testing graph I/O utilities...")
    
    # Save
    success = save_graph(test_graph, test_file)
    print(f"Save successful: {success}")
    
    # Load
    loaded_graph = load_graph(test_file)
    print(f"Load successful: {loaded_graph is not None}")
    
    # Get info
    if loaded_graph:
        info = get_graph_info(loaded_graph)
        print(f"Graph info: {info}")
    
    # Verify
    is_valid = verify_graph_file(test_file)
    print(f"File verification: {is_valid}")
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
        print("Test file cleaned up")
