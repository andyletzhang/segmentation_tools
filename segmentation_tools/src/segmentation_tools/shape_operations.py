import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union, polygonize, split
from skimage.draw import polygon

def get_enclosed_pixels(outline: np.ndarray) -> np.ndarray:
    """Return all pixels enclosed by the outline."""
    line = unary_union(LineString(outline))
    all_polygons = geoms_to_polygons(polygonize(line))

    enclosed_pixels = np.concatenate([enclosed_pixels_polygon(p) for p in all_polygons])
    return enclosed_pixels
        
def enclosed_pixels_polygon(shapely_polygon: Polygon) -> np.ndarray:
    poly_verts = np.array(shapely_polygon.exterior.coords)
    
    # Compute pixels by rasterizing the polygon
    rr, cc = polygon(poly_verts[:, 1], poly_verts[:, 0])
    
    polygon_pixels = np.column_stack((rr, cc))
    
    # Return as integer type
    return polygon_pixels.astype(np.int32)

def split_cell(outline: np.ndarray, curve: np.ndarray, min_area_threshold: int=0) -> list[np.ndarray]:
    cell_polygon = Polygon(outline)
    curve_line = LineString(curve)
   
    # Generate polygons
    polygons = geoms_to_polygons(split(cell_polygon, curve_line).geoms)
   
    if len(polygons) <= 1:  # just the original polygon
        return []
    
    # Filter out small polygons and merge them with neighbors
    if min_area_threshold > 0:
        polygons = merge_small_polygons(polygons, min_area_threshold)
        
    # If after merging we're left with only one polygon, return empty
    if len(polygons) <= 1:
        return []

    # Sort polygons by size and return
    return sorted([enclosed_pixels_polygon(poly) for poly in polygons], key=len, reverse=True)

def geoms_to_polygons(geoms) -> list[Polygon]:
    polygons=[]
    for geom in geoms:
        if isinstance(geom, Polygon):
            polygons.append(geom)
        else:
            polygons.extend(geom.geoms)
    return polygons

def merge_small_polygons(polygons: list[Polygon], min_area_threshold: int) -> list[Polygon]:
    """Merge small polygons with their largest neighbor"""
    # Sort polygons by area (largest first)
    sorted_polygons = sorted(polygons, key=lambda p: p.area, reverse=True)
    
    # Separate large and small polygons
    large_polygons = [p for p in sorted_polygons if p.area >= min_area_threshold]
    small_polygons = [p for p in sorted_polygons if p.area < min_area_threshold]
    
    # If no small polygons, return original list
    if not small_polygons:
        return sorted_polygons
    
    # If no large polygons, merge all small ones into the largest
    if not large_polygons:
        return [unary_union(small_polygons)] if small_polygons else []
    
    # For each small polygon, find the large polygon that shares the longest boundary
    for small_poly in small_polygons:
        best_match = None
        max_boundary_length = 0
        
        for large_poly in large_polygons:
            # Check if they share a boundary
            if small_poly.touches(large_poly):
                boundary_length = small_poly.boundary.intersection(large_poly.boundary).length
                if boundary_length > max_boundary_length:
                    max_boundary_length = boundary_length
                    best_match = large_poly
        
        # If no touching large polygon, find the nearest one
        if best_match is None:
            min_distance = float('inf')
            for large_poly in large_polygons:
                distance = small_poly.distance(large_poly)
                if distance < min_distance:
                    min_distance = distance
                    best_match = large_poly
        
        # Merge the small polygon with its best match
        idx = large_polygons.index(best_match)
        large_polygons[idx] = unary_union([best_match, small_poly])
    
    return large_polygons

def coords_to_mask(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = 1
    return mask

def get_mask_boundary(mask):
    from skimage.segmentation import find_boundaries

    boundaries = find_boundaries(mask, mode='inner')
    return boundaries

def get_bounding_box(cell_mask):
    # Find the rows and columns that contain True values
    rows = np.any(cell_mask, axis=1)
    cols = np.any(cell_mask, axis=0)

    # Find the indices of these rows and columns
    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]

    # Calculate the bounding box coordinates
    if row_indices.size and col_indices.size:
        min_row, max_row = row_indices[[0, -1]]
        min_col, max_col = col_indices[[0, -1]]
        return min_row, max_row + 1, min_col, max_col + 1
    else:
        # If no True values are found, raise
        return None, None, None, None