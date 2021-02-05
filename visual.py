import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from matplotlib.animation import PillowWriter, FuncAnimation
from nilearn import decomposition, plotting, image, input_data, surface
from nilearn.connectome import ConnectivityMeasure


class Visual:
    def __init__(self, angle=200, enable_3d=False):
        if enable_3d:
            self.fig = plt.figure(figsize=(30 / 2.54, 30 / 2.54))
            self.ax = self.fig.gca(projection='3d')
            self.ax.view_init(30, angle)
            self.ax.set_xlim(right=50 * 2)
            self.ax.set_ylim(top=50 * 2)
            self.ax.set_zlim(top=50 * 2)

    def show_slices(self, slices):
        """
        Function to display row of image slices
        :param slices: slice of fmri image
        """
        fig, axes = plt.subplots(1, len(slices))

        for i, slice in enumerate(slices):
            axes[i].imshow(slice.T, cmap="gray", origin="lower")

    def explode(self, data):
        shape_arr = np.array(data.shape)
        size = shape_arr[:3] * 2 - 1
        exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
        exploded[::2, ::2, ::2] = data
        return exploded

    def expand_coordinates(self, indices):
        x, y, z = indices
        x[1::2, :, :] += 1
        y[:, 1::2, :] += 1
        z[:, :, 1::2] += 1
        return x, y, z

    def normalize(self, arr):
        arr_min = np.min(arr)
        return (arr - arr_min) / (np.max(arr) - arr_min)

    def plot_cube(self, cube):
        cube = self.normalize(cube)

        facecolors = cm.coolwarm(cube)
        facecolors[:, :, :, -1] = cube
        facecolors = self.explode(facecolors)

        filled = facecolors[:, :, :, -1] != 0
        x, y, z = self.expand_coordinates(np.indices(np.array(filled.shape) + 1))

        self.ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)

    def generate_connectome(self, func, labels):
        canica = decomposition.CanICA(n_components=20, mask_strategy='background')
        canica.fit(func)
        # Retrieving the components
        components = canica.components_
        # Using a masker to project into the 3D space
        components_img = canica.masker_.inverse_transform(components)

        # Using a filter to extract the regions time series
        masker = input_data.NiftiMapsMasker(components_img, smoothing_fwhm=6,
                                            standardize=False, detrend=True,
                                            t_r=2.5, low_pass=0.1,
                                            high_pass=0.01)
        # Computing the regions signals and extracting the phenotypic information of interest
        subjects = []

        for func_file in func:
            time_series = masker.fit_transform(func_file)
            subjects.append(time_series)

        connectivity_biomarkers = {}

        kinds = ['correlation', 'partial correlation', 'tangent']
        for kind in kinds:
            conn_measure = ConnectivityMeasure(kind=kind, vectorize=True)
            connectivity_biomarkers[kind] = conn_measure.fit_transform(subjects)

        # For each kind, all individual coefficients are stacked in a unique 2D matrix.
        correlation_measure = ConnectivityMeasure(kind='correlation')
        correlation_matrices = correlation_measure.fit_transform(subjects)

        # Separating the correlation matrices between treatment and control subjects
        pain_correlations = []
        control_correlations = []
        for i in range(24):
            if labels[i] == 1:
                pain_correlations.append(correlation_matrices[i])
            else:
                control_correlations.append(correlation_matrices[i])

        return connectivity_biomarkers
