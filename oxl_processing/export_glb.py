import bpy
import sys

# Get command line arguments after "--"
argv = sys.argv
argv = argv[argv.index("--") + 1:]

if len(argv) < 1:
    print("Usage: blender --background file.blend --python export_glb.py -- output.glb")
    sys.exit()

export_file_path = argv[0]

# Deselect all objects
for obj in bpy.data.objects:
    obj.select_set(False)

# Select and process all mesh objects
mesh_objects = []
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        # Select the object
        obj.select_set(True)
        mesh_objects.append(obj)
        # Apply modifiers
        bpy.context.view_layer.objects.active = obj
        for modifier in obj.modifiers:
            try:
                bpy.ops.object.modifier_apply(modifier=modifier.name)
            except Exception as e:
                print(
                    f"Could not apply modifier {modifier.name} on {obj.name}: {e}")
        # Check UV maps
        mesh = obj.data
        if not mesh.uv_layers or any(len(uv_layer.data) == 0 for uv_layer in mesh.uv_layers):
            print(
                f"Object '{obj.name}' has missing or empty UV maps. Creating default UV map.")
            uv_layer = mesh.uv_layers.new(name="UVMap")
            # Unwrap the mesh
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.smart_project()
            bpy.ops.object.mode_set(mode='OBJECT')
        # Deselect the object after processing
        obj.select_set(False)

# Re-select the processed mesh objects
for obj in mesh_objects:
    obj.select_set(True)

# Ensure the active object is set
if mesh_objects:
    bpy.context.view_layer.objects.active = mesh_objects[0]
else:
    print("No mesh objects found to export.")
    sys.exit()


# Export to glTF (.glb)
bpy.ops.export_scene.gltf(
    filepath=export_file_path,
    export_format='GLB',       # Ensures binary .glb format
    use_selection=True         # Export only selected objects
)
