from pathlib import Path
import tempfile
import unittest

import numpy as np
import vtk

from organograph.mesh.OrganoidMesh import OrganoidMesh
from organograph.mesh.transform import transform_is_applied, transforms_equal


def sorted_points(points):
    points = np.asarray(points)
    return points[np.lexsort(points.T[::-1])]


def triangle_coordinates(mesh):
    return sorted(tuple(sorted(map(tuple, mesh.v[face]))) for face in mesh.f)


class MeshFormatNormalizationTests(unittest.TestCase):
    def test_stl_and_vtp_preserve_coordinates_and_share_normalization(self):
        vertices = np.array([[100., 200., 300.], [120., 200., 300.],
                             [100., 240., 300.], [100., 200., 360.]])
        faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
        points = vtk.vtkPoints()
        for vertex in vertices:
            points.InsertNextPoint(*vertex)
        cells = vtk.vtkCellArray()
        for face in faces:
            cells.InsertNextCell(3)
            for vertex in face:
                cells.InsertCellPoint(vertex)
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(cells)

        with tempfile.TemporaryDirectory() as tmp:
            vtp_path = str(Path(tmp) / "tetra.vtp")
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(vtp_path)
            writer.SetInputData(polydata)
            self.assertEqual(writer.Write(), 1)

            for binary in [False, True]:
                stl_path = str(Path(tmp) / f"tetra_{binary}.stl")
                writer = vtk.vtkSTLWriter()
                writer.SetFileName(stl_path)
                writer.SetInputData(polydata)
                if binary:
                    writer.SetFileTypeToBinary()
                else:
                    writer.SetFileTypeToASCII()
                self.assertEqual(writer.Write(), 1)

                for scale in [None, 5.0]:
                    with self.subTest(binary=binary, scale=scale):
                        vtp = OrganoidMesh(vtp_path)
                        stl = OrganoidMesh(stl_path)
                        for mesh in [vtp, stl]:
                            self.assertEqual(mesh.v.shape, (4, 3))
                            self.assertEqual(mesh.f.shape, (4, 3))
                            np.testing.assert_allclose(sorted_points(mesh.v), sorted_points(vertices))
                            self.assertFalse(transform_is_applied(mesh.coord_transform))
                        self.assertEqual(triangle_coordinates(stl), triangle_coordinates(vtp))

                        factor = 10.0 if scale is None else scale
                        expected = (vertices - vertices.mean(axis=0)) / factor
                        for mesh in [vtp, stl]:
                            center, actual_scale = (mesh.normalize_inplace() if scale is None
                                                    else mesh.normalize_inplace(scale=scale))
                            np.testing.assert_allclose(center, vertices.mean(axis=0))
                            self.assertEqual(actual_scale, factor)
                            np.testing.assert_allclose(sorted_points(mesh.v), sorted_points(expected))
                            self.assertTrue(transform_is_applied(mesh.coord_transform))
                        self.assertTrue(transforms_equal(stl.coord_transform, vtp.coord_transform))
                        self.assertEqual(triangle_coordinates(stl), triangle_coordinates(vtp))


if __name__ == "__main__":
    unittest.main()
