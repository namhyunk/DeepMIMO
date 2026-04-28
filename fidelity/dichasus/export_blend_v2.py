"""
Export INUE .blend to Mitsuba XML with manual PLY writing (guaranteed triangles).
"""

import bpy
import bmesh
import os
import struct

OUT_DIR = os.path.join(os.path.dirname(bpy.data.filepath), "inue_detailed")
MESH_DIR = os.path.join(OUT_DIR, "meshes")
os.makedirs(MESH_DIR, exist_ok=True)

mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
print(f"Found {len(mesh_objects)} mesh objects")

def write_ply(filepath, vertices, normals, faces):
    """Write a binary PLY file matching Sionna/Mitsuba expected format."""
    n_verts = len(vertices)
    n_faces = len(faces)

    with open(filepath, 'wb') as f:
        # Header - match simple scene format: (x,y,z,u,v) + int face indices
        header = f"ply\nformat binary_little_endian 1.0\n"
        header += f"element vertex {n_verts}\n"
        header += "property float x\nproperty float y\nproperty float z\n"
        header += "property float u\nproperty float v\n"
        header += f"element face {n_faces}\n"
        header += "property list uchar int vertex_indices\n"
        header += "end_header\n"
        f.write(header.encode('ascii'))

        # Vertices (x,y,z,u=0,v=0 - dummy UVs)
        for i in range(n_verts):
            v = vertices[i]
            f.write(struct.pack('<fffff', v[0], v[1], v[2], 0.0, 0.0))

        # Faces (all triangles, signed int indices)
        for face in faces:
            assert len(face) == 3, f"Non-triangle face: {face}"
            f.write(struct.pack('<B', 3))
            f.write(struct.pack('<iii', face[0], face[1], face[2]))

shapes = []
for obj in mesh_objects:
    name = obj.name.replace(" ", "_").replace(".", "_").lower()
    ply_path = os.path.join(MESH_DIR, f"{name}.ply")

    # Get evaluated mesh with modifiers
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)

    # Create bmesh and triangulate
    bm = bmesh.new()
    bm.from_mesh(eval_obj.data)

    # Apply object transform
    bm.transform(obj.matrix_world)

    # Triangulate
    bmesh.ops.triangulate(bm, faces=bm.faces[:])

    # Ensure normals
    bm.normal_update()

    # Extract data
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    vertices = [(v.co.x, v.co.y, v.co.z) for v in bm.verts]
    normals = [(v.normal.x, v.normal.y, v.normal.z) for v in bm.verts]
    faces = [(f.verts[0].index, f.verts[1].index, f.verts[2].index) for f in bm.faces]

    bm.free()

    # Write PLY
    write_ply(ply_path, vertices, normals, faces)

    # Get material
    mat_name = "default"
    if obj.material_slots:
        slot = obj.material_slots[0]
        if slot.material:
            mat_name = slot.material.name.replace(" ", "_").lower()

    shapes.append({'name': name, 'ply': f"meshes/{name}.ply", 'material': mat_name})
    print(f"  {name}: {len(vertices)} verts, {len(faces)} tri, mat={mat_name}")

# Material mapping
MATERIAL_MAP = {
    'concrete': {'rgb': '0.603814 0.603814 0.603814', 'roughness': '0.500000', 'id': 'mat-itu_concrete'},
    'plasterboard': {'rgb': '0.800000 0.800000 0.800000', 'roughness': '0.500000', 'id': 'mat-itu_plasterboard'},
    'ceiling': {'rgb': '0.367236 0.367236 0.367236', 'roughness': '0.500000', 'id': 'mat-itu_ceiling_board'},
    'floor': {'rgb': '0.603814 0.603814 0.603814', 'roughness': '0.500000', 'id': 'mat-itu_concrete'},
    'glass': {'rgb': '0.900000 0.900000 0.900000', 'roughness': '0.100000', 'id': 'mat-itu_glass'},
    'metal': {'rgb': '0.500000 0.500000 0.500000', 'roughness': '0.300000', 'id': 'mat-itu_metal'},
    'mirror': {'rgb': '0.900000 0.900000 0.900000', 'roughness': '0.050000', 'id': 'mat-itu_metal'},
    'wood': {'rgb': '0.450000 0.350000 0.250000', 'roughness': '0.600000', 'id': 'mat-itu_wood'},
    'antenna': {'rgb': '0.500000 0.500000 0.500000', 'roughness': '0.300000', 'id': 'mat-itu_metal'},
    'paper': {'rgb': '0.800000 0.800000 0.800000', 'roughness': '0.500000', 'id': 'mat-itu_plasterboard'},
    'default': {'rgb': '0.600000 0.600000 0.600000', 'roughness': '0.500000', 'id': 'mat-default'},
}

def get_material_props(mat_name):
    mat_lower = mat_name.lower()
    for key, props in MATERIAL_MAP.items():
        if key in mat_lower:
            return props
    return MATERIAL_MAP['default']

# Write XML
xml_path = os.path.join(OUT_DIR, "inue_detailed.xml")
with open(xml_path, 'w') as f:
    f.write('<scene version="2.1.0">\n\n')
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
    f.write('\t\t\t<matrix value="1 0 0 0 0 -1 0 0 0 0 -1 100 0 0 0 1"/>\n')
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

print(f"\nScene: {xml_path}")
print(f"Shapes: {len(shapes)}, Materials: {len(written_mats)}")
