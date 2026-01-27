#include <vector>

/**
 * TreeNode Class
 * 
 * Represents a single node in the RRT (Rapid Exploring Random Tree).
 * Each node contains:
 * - val: Position coordinates (x, y) for 2D or (x, y, z) for 3D
 * - parent: Pointer to the parent node (nullptr for root)
 * - children: List of child nodes that branch from this node
 * 
 * The tree structure allows us to build a path from start to goal by
 * connecting nodes through parent-child relationships.
 */
class TreeNode
{
public:
    std::vector<float> val;           // Position in configuration space
    TreeNode *parent;                 // Parent node (nullptr for root)
    std::vector<TreeNode *> children; // Child nodes
    
    // Constructor: Create a node at position 'val' with 'parent' as parent
    TreeNode(std::vector<float> val, TreeNode *parent) : val(val), parent(parent) {}
    
    // Add a child node to this node
    void add_child(TreeNode *child)
    {
        children.push_back(child);
    }
};
