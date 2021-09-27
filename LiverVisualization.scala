package project

import java.io.File

import liverfitting.exercise3.com.example.tet.LiverFitting016.{imgGroup, ui}
import scalismo.common.interpolation.BSplineImageInterpolator3D
import scalismo.io.{ImageIO, MeshIO, StatisticalModelIO}
import scalismo.mesh.TriangleMesh3D
import scalismo.ui.api.ScalismoUI

object LiverVisualization {


  def main(args: Array[String]) {

    // required to initialize native libraries (VTK, HDF5 ..)
    scalismo.initialize()

    val display = false

    // we need to random
    implicit val rng = scalismo.utils.Random(42)

    // Your application code goes below here. Below is a dummy application that reads a mesh and displays it

    // create a visualization window
    val ui = ScalismoUI()
    //val ui = ScalismoUIHeadless()

    // load reference mesh
    val LiverBroken = ui.createGroup("Liver")
    val mesh: TriangleMesh3D = MeshIO.readMesh(new File("datasets/testset/fitting-result/liver-016.stl")).get
    val mesh1: TriangleMesh3D = MeshIO.readMesh(new File("datasets/testset/fitting-result/liver-017.stl")).get
    val mesh2: TriangleMesh3D = MeshIO.readMesh(new File("datasets/testset/fitting-result/liver-018.stl")).get
    val mesh3: TriangleMesh3D = MeshIO.readMesh(new File("datasets/testset/fitting-result/liver-019.stl")).get
    val mesh4: TriangleMesh3D = MeshIO.readMesh(new File("datasets/testset/fitting-result/liver-020.stl")).get
    ui.show(LiverBroken, mesh, "Liver1")
   //ui.show(LiverBroken, mesh1, "Liver2")
    //ui.show(LiverBroken, mesh2, "Liver3")
    //ui.show(LiverBroken, mesh3, "Liver4")
    //ui.show(LiverBroken, mesh4, "Liver5")

    val discreteTargetImage = ImageIO.read3DScalarImageAsType[Short](new java.io.File("datasets/testset/volume-ct/liver-016.nii")).get

    val targetImage = discreteTargetImage.interpolate(BSplineImageInterpolator3D[Short](3))
    ui.show(LiverBroken, discreteTargetImage, "image")
    //val mesh1: TriangleMesh3D = MeshIO.readMesh(new File("data/new/partials/101156/101156_aligned_final.ply")).get
    //ui.show(LiverBroken, mesh1, "Liver")




  }
}
