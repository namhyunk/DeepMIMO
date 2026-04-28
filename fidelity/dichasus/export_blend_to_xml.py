"""
Export the full INUE .blend model to Mitsuba XML format for Sionna RT.
Uses Blender's Python API to iterate over mesh objects and export PLY files.
"""

import bpy
import os
import sys
import math
import bmesh
from mathutils import Matrix

# Output directory
OUT_DIR = os.path.join(os.path.dirname(bpy.data.filepath), "inue_detailed")
MESH_DIR = os.path.join(OUT_DIR, "meshes")
os.makedirs(MESH_DIR, exist_ok=True)

# Collect all mesh objects
mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
print(f"Found {len(mesh_objects)} mesh objects:")
for obj in mesh_objects:
    print(f"  {obj.name}: {len(obj.data.vertices)} verts, {len(obj.data.polygons)} faces")
    # Print material info
    for slot in obj.material_slots:
        if slot.material:
            mat = slot.material
            print(f"    Material: {mat.name}")

# Export each mesh as PLY
shapes = []
for obj in mesh_objects:
    name = obj.name.replace(" ", "_").replace(".", "_").lower()
    ply_path = os.path.join(MESH_DIR, f"{name}.ply")

    # Select only this object
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Apply transforms
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Triangulate using modifier (more reliable than bmesh for complex meshes)
    mod = obj.modifiers.new(name="Triangulate", type='TRIANGULATE')
    bpy.ops.object.modifier_apply(modifier=mod.name)

    # Verify all faces are triangles
    non_tri = [f for f in obj.data.polygons if len(f.vertices) != 3]
    if non_tri:
        print(f"    WARNING: {len(non_tri)} non-triangle faces remain in {obj.name}")

    # Export as PLY
    bpy.ops.wm.ply_export(filepath=ply_path,
                           export_selected_objects=True,
                           forward_axis='Y',
                           up_axis='Z',
                           export_normals=True)

    # Get material info
    mat_name = "itu_concrete"  # default
    if obj.material_slots:
        slot = obj.material_slots[0]
        if slot.material:
            mat_name = slot.material.name.replace(" ", "_").lower()

    shapes.append({
        'name': name,
        'ply': f"meshes/{name}.ply",
        'material': mat_name,
    })
    print(f"  Exported: {ply_path}")

# Generate Mitsuba XML
print(f"\nGenerating Mitsuba XML scene...")

# Map material names to ITU materials with principled BSDF parameters
# These are approximate - can be refined later
MATERIAL_MAP = {
    'concrete': {'rgb': '0.603814 0.603814 0.603814', 'roughness': '0.500000', 'id': 'mat-itu_concrete'},
    'plasterboard': {'rgb': '0.800000 0.800000 0.800000', 'roughness': '0.500000', 'id': 'mat-itu_plasterboard'},
    'ceiling': {'rgb': '0.367236 0.367236 0.367236', 'roughness': '0.500000', 'id': 'mat-itu_ceiling_board'},
    'glass': {'rgb': '0.900000 0.900000 0.900000', 'roughness': '0.100000', 'id': 'mat-itu_glass'},
    'metal': {'rgb': '0.500000 0.500000 0.500000', 'roughness': '0.300000', 'id': 'mat-itu_metal'},
    'wood': {'rgb': '0.450000 0.350000 0.250000', 'roughness': '0.600000', 'id': 'mat-itu_wood'},
    'default': {'rgb': '0.600000 0.600000 0.600000', 'roughness': '0.500000', 'id': 'mat-default'},
}

def get_material_props(mat_name):
    """Map Blender material name to ITU material properties."""
    mat_lower = mat_name.lower()
    for key, props in MATERIAL_MAP.items():
        if key in mat_lower:
            return props
    return MATERIAL_MAP['default']

# Collect unique materials
unique_materials = set()
for s in shapes:
    props = get_material_props(s['material'])
    unique_materials.add(props['id'])

# Write XML
xml_path = os.path.join(OUT_DIR, "inue_detailed.xml")
with open(xml_path, 'w') as f:
    f.write('<scene version="2.1.0">\n\n')

    # Integrator
    f.write('<!-- Camera and Rendering Parameters -->\n\n')
    f.write('\t<integrator type="path">\n')
    f.write('\t\t<integer name="max_depth" value="12"/>\n')
    f.write('\t</integrator>\n\n')

    # Materials
    f.write('<!-- Materials -->\n\n')
    written_mats = set()
    for s in shapes:
        props = get_material_props(s['material'])
        mat_id = props['id']
        if mat_id not in written_mats:
            written_mats.add(mat_id)
            f.write(f'\t<bsdf type="twosided" id="{mat_id}">\n')
            f.write('\t\t<bsdf type="principled">\n')
            f.write(f'\t\t\t<rgb value="{props["rgb"]}" name="base_color"/>\n')
            f.write('\t\t\t<float name="spec_tint" value="0.000000"/>\n')
            f.write('\t\t\t<float name="spec_trans" value="0.000000"/>\n')
            f.write('\t\t\t<float name="metallic" value="0.000000"/>\n')
            f.write('\t\t\t<float name="anisotropic" value="0.000000"/>\n')
            f.write(f'\t\t\t<float name="roughness" value="{props["roughness"]}"/>\n')
            f.write('\t\t\t<float name="sheen" value="0.000000"/>\n')
            f.write('\t\t\t<float name="sheen_tint" value="0.500000"/>\n')
            f.write('\t\t\t<float name="clearcoat" value="0.000000"/>\n')
            f.write('\t\t\t<float name="clearcoat_gloss" value="0.173205"/>\n')
            f.write('\t\t\t<float name="specular" value="0.500000"/>\n')
            f.write('\t\t</bsdf>\n')
            f.write('\t</bsdf>\n')

    # Emitter
    f.write('\n<!-- Emitters -->\n\n')
    f.write('\t<emitter type="directional" id="emit-Sun">\n')
    f.write('\t\t<rgb value="1.000000 1.000000 1.000000" name="irradiance"/>\n')
    f.write('\t\t<transform name="to_world">\n')
    f.write('\t\t\t<matrix value="1.000000 0.000000 0.000000 0.000000 ')
    f.write('0.000000 -1.000000 -0.000000 0.000000 ')
    f.write('0.000000 0.000000 -1.000000 100.000000 ')
    f.write('0.000000 0.000000 0.000000 1.000000"/>\n')
    f.write('\t\t</transform>\n')
    f.write('\t</emitter>\n\n')

    # Shapes
    f.write('<!-- Shapes -->\n\n')
    for s in shapes:
        props = get_material_props(s['material'])
        f.write(f'\t<shape type="ply" id="mesh-{s["name"]}">\n')
        f.write(f'\t\t<string name="filename" value="{s["ply"]}"/>\n')
        f.write('\t\t<boolean name="face_normals" value="true"/>\n')
        f.write(f'\t\t<ref id="{props["id"]}" name="bsdf"/>\n')
        f.write('\t</shape>\n')

    f.write('</scene>\n')

print(f"\nScene written to: {xml_path}")
print(f"Total shapes: {len(shapes)}")
print(f"Total materials: {len(written_mats)}")
