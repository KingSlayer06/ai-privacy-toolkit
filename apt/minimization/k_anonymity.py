"""
K-Anonymity Protection Module

This module implements k-anonymity enforcement to protect against homogeneity attacks.
K-anonymity ensures that each generalized record is indistinguishable from at least
k-1 other records, preventing attackers from inferring sensitive information through
cluster membership analysis.

Security Purpose:
- Protects against homogeneity attacks where attackers exploit small clusters
- Ensures each generalized record has sufficient anonymity set
- Provides formal privacy guarantee through k-anonymity property
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np


class KAnonymityEnforcer:
    """
    Enforces k-anonymity constraints on generalization cells.
    
    K-anonymity requires that each equivalence class (cell) contains at least k records.
    This prevents homogeneity attacks where an attacker could infer sensitive information
    by identifying which cluster a record belongs to.
    
    Security Mechanism:
    The enforcer monitors cell sizes and merges cells that violate k-anonymity constraints.
    This ensures that even if an attacker knows a record's cluster membership, they cannot
    uniquely identify the individual or infer sensitive attributes with high confidence.
    
    :param k: Minimum number of records required per equivalence class.
              Must be at least 2. Higher values provide stronger privacy but may reduce utility.
    :type k: int
    """
    
    def __init__(self, k: int = 2):
        """
        Initialize the k-anonymity enforcer.
        
        :param k: Minimum records per equivalence class
        :type k: int
        """
        if k < 2:
            raise ValueError("k must be at least 2 for meaningful k-anonymity protection")
        
        self.k = k
        self.violation_count = 0
        self.merge_count = 0
        
        print(f"Initialized with k={k}")
        print(f"  Each equivalence class must contain at least {k} records")
    
    def enforce_k_anonymity(self, cells: List[Dict], samples: pd.DataFrame, 
                           cells_by_id: Dict) -> tuple[List[Dict], Dict]:
        """
        Enforce k-anonymity by merging cells that violate the constraint.
        
        This function checks each cell's size and merges cells that contain fewer
        than k records. Merging is done by combining cells with similar characteristics
        (e.g., adjacent cells in the decision tree).
        
        Security Purpose:
        Prevents homogeneity attacks by ensuring no cell is too small. Even if an
        attacker identifies a record's cluster, they cannot uniquely identify the
        individual or infer sensitive attributes.
        
        Function Call Sequence:
        Called after _calculate_cells() and _modify_cells() in fit(), and also
        during tree pruning iterations to maintain k-anonymity throughout the process.
        
        :param cells: List of cell dictionaries containing generalization information
        :type cells: List[Dict]
        :param samples: DataFrame containing the data samples
        :type samples: pd.DataFrame
        :param cells_by_id: Dictionary mapping cell IDs to cell dictionaries
        :type cells_by_id: Dict
        :return: Tuple of (updated_cells, updated_cells_by_id) with k-anonymity enforced
        :rtype: tuple[List[Dict], Dict]
        """
        print(f"Checking k-anonymity compliance (k={self.k})...")
        
        # Count records per cell
        cell_counts = self._count_records_per_cell(cells, samples)
        
        # Identify violations
        violations = self._identify_violations(cells, cell_counts)
        
        if not violations:
            print(f"All cells satisfy k-anonymity (k={self.k})")
            return cells, cells_by_id
        
        print(f"Found {len(violations)} cells violating k-anonymity")
        self.violation_count += len(violations)
        
        # Merge violating cells
        merged_cells, merged_cells_by_id = self._merge_violating_cells(
            cells, cells_by_id, violations, cell_counts, samples
        )
        
        self.merge_count += len(violations)
        print(f"Merged {len(violations)} cells to enforce k-anonymity")
        
        return merged_cells, merged_cells_by_id
    
    def _count_records_per_cell(self, cells: List[Dict], samples: pd.DataFrame) -> Dict[int, int]:
        """
        Count how many records belong to each cell by checking which records match each cell's generalization
        constraints.
        
        :param cells: List of cell dictionaries
        :type cells: List[Dict]
        :param samples: DataFrame containing data samples
        :type samples: pd.DataFrame
        :return: Dictionary mapping cell IDs to record counts
        """
        cell_counts = {}
        # Use positional index instead of DataFrame index to avoid index mismatch
        mapped = np.zeros(len(samples), dtype=bool)
        
        for cell in cells:
            count = 0
            for pos_idx, (_, row) in enumerate(samples.iterrows()):
                if not mapped[pos_idx] and self._cell_contains_record(cell, row):
                    count += 1
                    mapped[pos_idx] = True
            cell_counts[cell['id']] = count
        
        return cell_counts
    
    def _cell_contains_record(self, cell: Dict, row: pd.Series) -> bool:
        """
        Check if a record matches a cell's generalization constraints.
        
        :param cell: Cell dictionary with ranges and categories
        :type cell: Dict
        :param row: Data record
        :type row: pd.Series
        :return: True if record matches cell constraints, False otherwise
        """
        # Check numeric ranges
        for feature, range_dict in cell.get('ranges', {}).items():
            if feature in row.index:
                value = row[feature]
                if range_dict.get('start') is not None and value <= range_dict['start']:
                    return False
                if range_dict.get('end') is not None and value > range_dict['end']:
                    return False
        
        # Check categorical categories
        for feature, categories in cell.get('categories', {}).items():
            if feature in row.index:
                value = row[feature]
                if value not in categories:
                    return False
        
        return True
    
    def _identify_violations(self, cells: List[Dict], cell_counts: Dict[int, int]) -> List[int]:
        """
        Identify cells that violate k-anonymity.
        
        :param cells: List of cell dictionaries
        :type cells: List[Dict]
        :param cell_counts: Dictionary mapping cell IDs to record counts
        :type cell_counts: Dict[int, int]
        :return: List of cell IDs that violate k-anonymity
        """
        violations = []
        for cell in cells:
            cell_id = cell['id']
            count = cell_counts.get(cell_id, 0)
            if count < self.k:
                violations.append(cell_id)
                print(f"  Cell {cell_id}: {count} records (requires at least {self.k})")
        
        return violations
    
    def _merge_violating_cells(self, cells: List[Dict], cells_by_id: Dict,
                               violations: List[int], cell_counts: Dict[int, int],
                               samples: pd.DataFrame) -> tuple[List[Dict], Dict]:
        """
        Merge cells that violate k-anonymity with their most similar neighbors.
        
        Merging strategy: Combine violating cells with adjacent cells (siblings in
        the decision tree) or cells with similar generalization patterns.
        
        :param cells: List of cell dictionaries
        :type cells: List[Dict]
        :param cells_by_id: Dictionary mapping cell IDs to cell dictionaries
        :type cells_by_id: Dict
        :param violations: List of cell IDs violating k-anonymity
        :type violations: List[int]
        :param cell_counts: Dictionary mapping cell IDs to record counts
        :type cell_counts: Dict[int, int]
        :param samples: DataFrame containing data samples
        :type samples: pd.DataFrame
        :return: Tuple of (merged_cells, merged_cells_by_id)
        """
        merged_cells = []
        merged_cells_by_id = {}
        violation_set = set(violations)
        
        # Group violating cells for merging
        merged_groups = []
        remaining_violations = set(violations)
        
        for cell in cells:
            cell_id = cell['id']
            
            if cell_id in violation_set:
                # Find a suitable cell to merge with
                merge_target = self._find_merge_target(
                    cell, cells, cells_by_id, violation_set, cell_counts
                )
                
                if merge_target:
                    # Merge cells
                    merged_cell = self._merge_two_cells(cell, merge_target)
                    merged_groups.append((cell_id, merge_target['id'], merged_cell))
                    remaining_violations.discard(cell_id)
                    remaining_violations.discard(merge_target['id'])
                else:
                    # No suitable merge target found, keep cell but mark for later handling
                    merged_cells.append(cell)
                    merged_cells_by_id[cell_id] = cell
            elif cell_id not in [mg[1] for mg in merged_groups]:
                # Cell is not a violation and not already merged
                merged_cells.append(cell)
                merged_cells_by_id[cell_id] = cell
        
        # Add merged cells
        for _, _, merged_cell in merged_groups:
            merged_cells.append(merged_cell)
            merged_cells_by_id[merged_cell['id']] = merged_cell
        
        # Handle remaining violations by merging with largest available cell
        for violation_id in remaining_violations:
            if violation_id in cells_by_id:
                violation_cell = cells_by_id[violation_id]
                # Find largest non-violating cell
                largest_cell = max(
                    [c for c in merged_cells if c['id'] not in violation_set],
                    key=lambda c: cell_counts.get(c['id'], 0),
                    default=None
                )
                
                if largest_cell:
                    merged_cell = self._merge_two_cells(violation_cell, largest_cell)
                    # Replace largest cell with merged cell
                    merged_cells = [c for c in merged_cells if c['id'] != largest_cell['id']]
                    merged_cells.append(merged_cell)
                    merged_cells_by_id[merged_cell['id']] = merged_cell
                    del merged_cells_by_id[largest_cell['id']]
        
        return merged_cells, merged_cells_by_id
    
    def _find_merge_target(self, violating_cell: Dict, cells: List[Dict],
                          cells_by_id: Dict, violation_set: set,
                          cell_counts: Dict[int, int]) -> Optional[Dict]:
        """
        Find the best cell to merge with a violating cell.
        
        Strategy: Prefer merging with adjacent cells (siblings in the decision tree) or cells
        with similar generalization patterns.
        
        :param violating_cell: Cell that violates k-anonymity
        :type violating_cell: Dict
        :param cells: List of all cells
        :type cells: List[Dict]
        :param cells_by_id: Dictionary mapping cell IDs to cells
        :type cells_by_id: Dict
        :param violation_set: Set of violating cell IDs
        :type violation_set: set
        :param cell_counts: Dictionary mapping cell IDs to record counts
        :type cell_counts: Dict[int, int]
        :return: Best cell to merge with, or None if no suitable target found
        """
        # Prefer non-violating cells with similar characteristics
        best_target = None
        best_score = -1
        
        for cell in cells:
            if cell['id'] == violating_cell['id']:
                continue
            
            # Calculate similarity score between violating cell and candidate cell
            similarity = self._calculate_cell_similarity(violating_cell, cell)
            count = cell_counts.get(cell['id'], 0)
            
            # Prefer non-violating cells, but allow merging violations together
            if cell['id'] not in violation_set:
                score = similarity * 2 + count
            else:
                score = similarity + count
            
            if score > best_score:
                best_score = score
                best_target = cell
        
        return best_target
    
    def _calculate_cell_similarity(self, cell1: Dict, cell2: Dict) -> float:
        """
        Calculate similarity score between two cells.
        
        Higher score indicates cells are more similar and better candidates for merging.
        
        :param cell1: First cell
        :type cell1: Dict
        :param cell2: Second cell
        :type cell2: Dict
        :return: Similarity score (0.0 to 1.0)
        """
        common_features = 0
        total_features = 0
        
        # Check ranges
        ranges1 = set(cell1.get('ranges', {}).keys())
        ranges2 = set(cell2.get('ranges', {}).keys())
        common_features += len(ranges1 & ranges2)
        total_features += len(ranges1 | ranges2)
        
        # Check categories
        cats1 = set(cell1.get('categories', {}).keys())
        cats2 = set(cell2.get('categories', {}).keys())
        common_features += len(cats1 & cats2)
        total_features += len(cats1 | cats2)
        
        if total_features == 0:
            return 0.0
        
        return common_features / total_features
    
    def _merge_two_cells(self, cell1: Dict, cell2: Dict) -> Dict:
        """
        Merge two cells into a single cell with combined generalizations.
        
        The merged cell contains the union of ranges and categories from both cells,
        ensuring that all records from both cells match the merged cell.
        
        :param cell1: First cell to merge
        :type cell1: Dict
        :param cell2: Second cell to merge
        :type cell2: Dict
        :return: Merged cell 
        """
        merged_cell = {
            'id': cell1['id'],  # Use first cell's ID
            'ranges': {},
            'categories': {},
            'untouched': list(set(cell1.get('untouched', []) + cell2.get('untouched', []))),
            'label': cell1.get('label'),  # Use first cell's label
            'hist': cell1.get('hist', []) + cell2.get('hist', []),  # Combine histograms
            'representative': cell1.get('representative', {})
        }
        
        # Merge ranges by taking union
        all_range_features = set(cell1.get('ranges', {}).keys()) | set(cell2.get('ranges', {}).keys())
        for feature in all_range_features:
            range1 = cell1.get('ranges', {}).get(feature, {'start': None, 'end': None})
            range2 = cell2.get('ranges', {}).get(feature, {'start': None, 'end': None})
            
            merged_range = {
                'start': min(
                    r for r in [range1.get('start'), range2.get('start')] if r is not None
                ) if any(r is not None for r in [range1.get('start'), range2.get('start')]) else None,
                'end': max(
                    r for r in [range1.get('end'), range2.get('end')] if r is not None
                ) if any(r is not None for r in [range1.get('end'), range2.get('end')]) else None
            }
            merged_cell['ranges'][feature] = merged_range
        
        # Merge categories by taking union
        all_cat_features = set(cell1.get('categories', {}).keys()) | set(cell2.get('categories', {}).keys())
        for feature in all_cat_features:
            cats1 = set(cell1.get('categories', {}).get(feature, []))
            cats2 = set(cell2.get('categories', {}).get(feature, []))
            merged_cell['categories'][feature] = list(cats1 | cats2)
        
        return merged_cell
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about k-anonymity enforcement.
        
        :return: Dictionary with violation and merge statistics
        """
        return {
            'k': self.k,
            'violation_count': self.violation_count,
            'merge_count': self.merge_count
        }
