from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import numpy as np
import torch
from monai.bundle import ConfigParser, download
from monai.transforms import LoadImage, EnsureChannelFirst, Orientation, Compose
from skimage import measure
import vtk

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define your model and preprocessing steps here

@app.route('/process', methods=['POST'])
def process_scan():
    # Receive the NIfTI file from the client
    file = request.files['file']
    input_path = os.path.join('/tmp', file.filename)
    file.save(input_path)

    # Perform segmentation and save the model
    output_directory = '/tmp'
    output_path = os.path.join(output_directory, 'segmented_model.obj')
    segment_and_save_model(input_path, output_directory)

    # Provide the path to the generated OBJ file
    return jsonify({'output': output_path})

def segment_and_save_model(input_path, output_directory):
    image_loader = LoadImage(image_only=True)
    CT = image_loader(input_path)

    preprocessing_pipeline = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes='LPS')
    ])
    CT = preprocessing_pipeline(input_path)

    datadir = "/Users/raja/Downloads/dataset_output_2/train/images"
    model_name = "wholeBody_ct_segmentation"
    download(name=model_name, bundle_dir=datadir)
    model_path = os.path.join(datadir, 'wholeBody_ct_segmentation', 'models', 'model_lowres.pt')
    config_path = os.path.join(datadir, 'wholeBody_ct_segmentation', 'configs', 'inference.json')

    config = ConfigParser()
    config.read_config(config_path)

    preprocessing = config.get_parsed_content("preprocessing")
    data = preprocessing({'image': input_path})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = config.get_parsed_content("network")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    inferer = config.get_parsed_content("inferer")
    postprocessing = config.get_parsed_content("postprocessing")

    with torch.no_grad():
        data['pred'] = inferer(data['image'].unsqueeze(0).to(device), network=model)
    data['pred'] = data['pred'][0]
    data['image'] = data['image'][0]
    data = postprocessing(data)

    # Visualize and save the segmentation result
    visualize_3d_multilabel(data['pred'], output_directory)

def visualize_3d_multilabel(segmentation, output_directory):
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    if segmentation.ndim == 4 and segmentation.shape[0] == 1:
        segmentation = np.squeeze(segmentation, axis=0)

    if segmentation.ndim != 3:
        raise ValueError(f'Input volume should be a 3D numpy array. Got shape: {segmentation.shape}')

    segmentation = np.asarray(segmentation, dtype=np.uint8)

    color_map = {
        1: [0, 1, 0],           # Spleen - Green
        2: [0, 0, 1],           # Right Kidney - Blue
        3: [1, 0, 0],           # Left Kidney - Red
        4: [1, 1, 0],           # Gallbladder - Yellow
        5: [1, 0, 1],           # Liver - Magenta
        6: [0, 1, 1],           # Stomach - Cyan
        7: [0.5, 0, 0.5],       # Aorta - Purple
        8: [1, 0.5, 0],         # Inferior Vena Cava - Orange
        9: [0, 0.5, 0.5],       # Portal Vein and Splenic Vein - Teal
        10: [0.5, 0.5, 0],      # Pancreas - Olive
        11: [0.5, 0, 0],        # Adrenal Gland Right - Maroon
        12: [0, 0.5, 0],        # Adrenal Gland Left - Lime
        13: [0.5, 0.5, 0.5],    # Vertebrae - Gray
        14: [0.8, 0.8, 0.8],    # Ribs - Light Gray
        15: [0.7, 0.2, 0.7],    # Gluteal Muscles - Purple
        55: [0.2, 0.8, 0.2],    # Small Bowel - Light Green
        56: [0.8, 0.2, 0.2],    # Duodenum - Light Red
        57: [0.2, 0.2, 0.8]     # Colon - Light Blue
    }

    vertebrae_labels = list(range(18, 42))  # L5 to C1
    rib_labels = list(range(58, 82))        # Left and Right ribs 1 to 12
    gluteal_labels = [94, 95, 96, 97, 98, 99]  # Gluteal muscles

    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetAlphaBitPlanes(1)  # Enable alpha channel
    render_window.SetMultiSamples(0)    # Disable multisampling
    render_window.SetNumberOfLayers(2)  # Use multiple layers

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    combined_mesh = vtk.vtkAppendPolyData()

    for label_idx in np.unique(segmentation):
        if label_idx == 0:
            continue

        verts, faces, _, _ = measure.marching_cubes(segmentation == label_idx, level=0.5)
        mesh = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        triangles = vtk.vtkCellArray()

        for i, vert in enumerate(verts):
            points.InsertNextPoint(vert)
        for face in faces:
            triangle = vtk.vtkTriangle()
            for j in range(3):
                triangle.GetPointIds().SetId(j, face[j])
            triangles.InsertNextCell(triangle)

        mesh.SetPoints(points)
        mesh.SetPolys(triangles)

        combined_mesh.AddInputData(mesh)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(mesh)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        if label_idx in vertebrae_labels:
            color = color_map[13]
        elif label_idx in rib_labels:
            color = color_map[14]
        elif label_idx in gluteal_labels:
            color = color_map[15]
        else:
            color = color_map.get(label_idx, [1, 1, 1])  # Default to white if label not in color_map

        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(1.0)

        renderer.AddActor(actor)

    combined_mesh.Update()
    output_path = os.path.join(output_directory, 'combined_model.obj')
    save_vtk_polydata(combined_mesh.GetOutput(), output_path)

    renderer.SetBackground(0, 0, 0)  # Background color in RGB (black)
    renderer.SetBackgroundAlpha(0.0)  # Set the background alpha (transparency)

    render_window.SetSize(800, 800)

    interactor.Initialize()
    render_window.Render()
    interactor.Start()

def save_vtk_polydata(polydata, filename):
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
