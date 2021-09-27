package liverfitting.exercise3

package com.example.tet

import java.awt.Color
import java.io.File

import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.common.PointId
import scalismo.common.interpolation.BSplineImageInterpolator3D
import scalismo.geometry.{EuclideanVector, Landmark, Point, _3D}
import scalismo.image.DiscreteScalarImage
import scalismo.io.{ActiveShapeModelIO, ImageIO, LandmarkIO, MeshIO, StatisticalModelIO}
import scalismo.mesh.{MeshMetrics, TriangleMesh, TriangleMesh3D}
import scalismo.sampling.algorithms.MetropolisHastings
import scalismo.sampling.evaluators.ProductEvaluator
import scalismo.sampling.loggers.AcceptRejectLogger
import scalismo.sampling.proposals.MixtureProposal
import scalismo.statisticalmodel.{MultivariateNormalDistribution, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random
import liverfitting.evaluators.{CachedEvaluator, CorrespondenceEvaluator, PriorEvaluator}
import liverfitting.logger.Logger
import liverfitting.parameters.{Parameters, Sample}
import liverfitting.proposals.{RotationUpdateProposal, ShapeUpdateProposal, TranslationUpdateProposal}
import liverfitting.utils.DiagnosticPlots
import scalismo.statisticalmodel.asm.ActiveShapeModel
import scalismo.geometry._
import scalismo.ui.api._
import scalismo.registration._
import scalismo.statisticalmodel.asm._
import scalismo.io.{ActiveShapeModelIO, ImageIO}
import breeze.linalg.DenseVector
import liverfitting.tutorial.Tutorial15.marginalizeModelForCorrespondences
import scalismo.sampling.DistributionEvaluator

object LiverFitting016 {

  val ui = ScalismoUI()
 // val ui = ScalismoUIHeadless()

  val modelGroup = ui.createGroup("model")
  val imgGroup = ui.createGroup("image")
  //val modelGroup1 = ui.createGroup("model1")
 // val imgGroup1 = ui.createGroup("image1")



  def computeCenterOfMass(mesh : TriangleMesh[_3D]) : Point[_3D] = {
    val normFactor = 1.0 / mesh.pointSet.numberOfPoints
    mesh.pointSet.points.foldLeft(Point(0, 0, 0))((sum, point) => sum + point.toVector * normFactor)
  }


  def fitMesh(activeShapeModel : ActiveShapeModel,
              correspondencePoints : Seq[(PointId, Point[_3D])],
              contourPoints : Seq[Point[_3D]], image : DiscreteScalarImage[_3D, Float], useStoredPoseFit : Boolean = false)
             (implicit rng : scalismo.utils.Random) : Unit = {

    val shapeModel = activeShapeModel.statisticalModel

    val modelView = ui.show(modelGroup, shapeModel, "volume model")
    val gpTrans = modelView.shapeModelTransformationView.shapeTransformationView
    val poseTrans = modelView.shapeModelTransformationView.poseTransformationView

    // your code comes here
   // ???

    var landmarkNoiseVariance = 3.0
    var uncertainty = MultivariateNormalDistribution(
      DenseVector.zeros[Double](3),
      DenseMatrix.eye[Double](3) * landmarkNoiseVariance
    )


    var correspondences = correspondencePoints.map(corr => {
      val (referenceID, targetPoint) =  corr
      (referenceID, targetPoint, uncertainty)
    })



    var likelihoodEvaluator = CachedEvaluator(CorrespondenceEvaluator(shapeModel, correspondences))
    var priorEvaluator = CachedEvaluator(PriorEvaluator(shapeModel))

    var posteriorEvaluator = ProductEvaluator(priorEvaluator, likelihoodEvaluator)

    var shapeUpdateProposal = ShapeUpdateProposal(shapeModel.rank, 0.1)
    var rotationUpdateProposal = RotationUpdateProposal(0.01)
    var translationUpdateProposal = TranslationUpdateProposal(1.0)
    var generator = MixtureProposal.fromProposalsWithTransition(
      (0.01, shapeUpdateProposal),   //at 0.6, distance is 277.646, 0.2 279.0666, 0.9 276.7218
      (0.2, rotationUpdateProposal),//initial 0.2
      (0.2, translationUpdateProposal),
      (0.002, shapeUpdateProposal),   //at 0.6, distance is 277.646, 0.2 279.0666, 0.9 276.7218
      (0.2, rotationUpdateProposal),//initial 0.2
      (0.2, translationUpdateProposal)
    )



    var initialParameters = Parameters(
      EuclideanVector(0, 0, 0),
      (0.0, 0.0, 0.0),
      DenseVector.zeros[Double](shapeModel.rank)
    )


    var initialSample = Sample("initial", initialParameters, computeCenterOfMass(shapeModel.mean))

    var chain = MetropolisHastings(generator, posteriorEvaluator)
    var logger = new Logger()
    var mhIterator = chain.iterator(initialSample, logger)

    var samplingIterator = for((sample, iteration) <- mhIterator.zipWithIndex) yield {
      println("iteration " + iteration)
      if (iteration % 500 == 0) {
        modelView.shapeModelTransformationView.shapeTransformationView.coefficients = sample.parameters.modelCoefficients
        modelView.shapeModelTransformationView.poseTransformationView.transformation = sample.poseTransformation
      }
      sample
    }


    var samples = samplingIterator.drop(1000).take(25000).toIndexedSeq
    println(logger.acceptanceRatios())

   var bestSample = samples.maxBy(posteriorEvaluator.logValue)
    println("best sample"+bestSample)
   var bestFit = shapeModel.instance(bestSample.parameters.modelCoefficients).transform(bestSample.poseTransformation)
    println()
    var resultGroup = ui.createGroup("result")
    var final_fit=ui.show(resultGroup, bestFit, "best fit")
    final_fit.color=Color.RED
    val COM= computeCenterOfMass(bestFit);
    println(COM)
    //val referenceMesh = MeshIO.readMesh(new java.io.File("datasets/validationset/fitting-result/liver-orig011.vtk")).get

    val modelGroup_ref = ui.createGroup("gp-model")
   // val referenceView = ui.show(modelGroup_ref, referenceMesh, "reference")
    //val distance=MeshMetrics.hausdorffDistance(shapeModel.mean,bestFit);
    //println("Distance is "+ distance)
    //val filename=new File("Data/liver_test.stl")
   // MeshIO.writeMesh(bestFit, filename)

    //val profiles = shapeModel.profiles


    val model=shapeModel
//    def marginalizeModelForCorrespondences(model: StatisticalMeshModel,
//                                           correspondences: Seq[(PointId, Point[_3D], MultivariateNormalDistribution)])
//    : (StatisticalMeshModel, Seq[(PointId, Point[_3D], MultivariateNormalDistribution)]) = {
//
//      val (modelIds, _, _) = correspondences.unzip3
//      val marginalizedModel = model.marginal(modelIds.toIndexedSeq)
//      val newCorrespondences = correspondences.map(idWithTargetPoint => {
//        val (id, targetPoint, uncertainty) = idWithTargetPoint
//        val modelPoint = model.referenceMesh.pointSet.point(id)
//        val newId = marginalizedModel.referenceMesh.pointSet.findClosestPoint(modelPoint).id
//        (newId, targetPoint, uncertainty)
//      })
//      (marginalizedModel, newCorrespondences)
//    }


    //val (marginalizedModel, newCorrespondences) = marginalizeModelForCorrespondences(model, correspondences)

    val discreteTargetImage1 = ImageIO.read3DScalarImageAsType[Float](new java.io.File("datasets/validationset/volume-ct/liver-016.nii")).get.map(_.toFloat)


    val preprocessedImage = activeShapeModel.preprocessor(discreteTargetImage1)

    //val v_likelihoodForMesh=likelihoodForMesh(activeShapeModel, bestFit, preprocessedImage)




     landmarkNoiseVariance = 6.0
    uncertainty = MultivariateNormalDistribution(
      DenseVector.zeros[Double](3),
      DenseMatrix.eye[Double](3) * landmarkNoiseVariance
    )

     correspondences = correspondencePoints.map(corr => {
      val (referenceID, targetPoint) =  corr
      (referenceID, targetPoint, uncertainty)
    })


    case class ImageEvaluator(asm : ActiveShapeModel, preprocessedImage: PreprocessedImage)
      extends DistributionEvaluator[Sample] {
      override def logValue(sample: Sample): Double = {


        val ids = asm.profiles.ids
        // println(ids)

        val likelihoods = for (id <- ids) yield {
          val profile = asm.profiles(id)
          //val mesh: TriangleMesh3D = MeshIO.readMesh(new File("datasets/validationset/fitting-result/liver-orig011.vtk")).get
          //val mesh = StatisticalModelIO.readStatisticalMeshModel(new java.io.File("data/test_2aligned.h5")).get
          //val mesh = asm.instance(sample.parameters.modelCoefficients).transform(sample.poseTransformation)
          val mesh = model.instance(sample.parameters.modelCoefficients).transform(sample.poseTransformation)




          val profilePointOnMesh = mesh.pointSet.point(profile.pointId)
          // println(profilePointOnMesh)
          //  println(profile.pointId)

          val featureAtPoint = asm.featureExtractor(preprocessedImage, profilePointOnMesh, mesh, profile.pointId).get
          profile.distribution.logpdf(featureAtPoint)
        }
        likelihoods.sum


      }
    }

      likelihoodEvaluator = CachedEvaluator(ImageEvaluator(activeShapeModel, preprocessedImage))
      priorEvaluator = CachedEvaluator(PriorEvaluator(shapeModel))

     posteriorEvaluator = ProductEvaluator(priorEvaluator, likelihoodEvaluator)

     shapeUpdateProposal = ShapeUpdateProposal(shapeModel.rank, 0.1)
     rotationUpdateProposal = RotationUpdateProposal(0.01)
     translationUpdateProposal = TranslationUpdateProposal(1.0)
     generator = MixtureProposal.fromProposalsWithTransition(
      (0.2, shapeUpdateProposal),   //at 0.6, distance is 277.646, 0.2 279.0666, 0.9 276.7218
      (0.2, rotationUpdateProposal),//initial 0.2
      (0.2, translationUpdateProposal)

    )

     initialParameters = Parameters(
      EuclideanVector(0,0,0),
      (0,0,0),
      DenseVector.zeros[Double](shapeModel.rank)
    )


     initialSample = bestSample

println ("i am here $")
    chain = MetropolisHastings(generator, posteriorEvaluator)
     logger = new Logger()
      mhIterator = chain.iterator(initialSample, logger)
    println ("i am here $$")

      samplingIterator = for((sample, iteration) <- mhIterator.zipWithIndex) yield {
      println("iteration " + iteration)
      if (iteration % 500 == 0) {
        modelView.shapeModelTransformationView.shapeTransformationView.coefficients = sample.parameters.modelCoefficients
        modelView.shapeModelTransformationView.poseTransformationView.transformation = sample.poseTransformation
        println ("i am here $$$")
      }
      sample
    }


     samples = samplingIterator.drop(1000).take(10000).toIndexedSeq
    println(logger.acceptanceRatios())


    println ("i am here $$$$")



   bestSample = samples.maxBy(posteriorEvaluator.logValue)
var bestFit1 = shapeModel.instance(bestSample.parameters.modelCoefficients).transform(bestSample.poseTransformation)
    println ("i am here $$$$$")

   // val resultGroup = ui.createGroup("result")
    val final_fit1=ui.show(resultGroup, bestFit1, "best fit1")
    final_fit1.color=Color.GREEN

    val distance=MeshMetrics.hausdorffDistance(shapeModel.mean,bestFit1);
    println("Distance is "+ distance)
    val filename=new File("datasets/testset/fitting-result/liver-016.stl")
    MeshIO.writeMesh(bestFit1, filename)

   // val referencePoints : Seq[Point[_3D]] = correspondences.map(lm => lm.point)



/*
    def computeMean(model : StatisticalMeshModel, id: PointId): Point[_3D] = {
      var mean = EuclideanVector(0, 0, 0)
      for (sample <- samples) yield {
        val modelInstance = model.instance(sample.parameters.modelCoefficients)
        val pointForInstance = modelInstance.transform(sample.poseTransformation).pointSet.point(id)
        mean += pointForInstance.toVector
      }
      (mean * 1.0 / samples.size).toPoint
    }

    def computeCovarianceFromSamples(model : StatisticalMeshModel, id: PointId, mean: Point[_3D]): SquareMatrix[_3D] = {
      var cov = SquareMatrix.zeros[_3D]
      for (sample <- samples) yield {
        val modelInstance = model.instance(sample.parameters.modelCoefficients)
        val pointForInstance = modelInstance.transform(sample.poseTransformation).pointSet.point(id)
        val v = pointForInstance - mean
        cov += v.outer(v)
      }
      cov * (1.0 / samples.size)
    }


    for ((id, _, _) <- newCorrespondences) {
      val meanPointPosition = computeMean(marginalizedModel, id)
      println(s"expected position for point at id $id  = $meanPointPosition")
      val cov = computeCovarianceFromSamples(marginalizedModel, id, meanPointPosition)
      println(s"posterior variance computed  for point at id (shape and pose) $id  = ${cov(0,0)}, ${cov(1,1)}, ${cov(2,2)}")
    }





*/







}


  def main(args : Array[String]) : Unit = {
    scalismo.initialize()
    implicit val rng = scalismo.utils.Random(42)

    val activeShapeModel = ActiveShapeModelIO.readActiveShapeModel(new java.io.File("./datasets/asm/liver-asm.h5")).get
    val shapeModel = activeShapeModel.statisticalModel
    println(shapeModel.rank)

    val modelLandmarkPoints = LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/model.json"))
      .get.map(lm => lm.point)
    val targetLandmarkPoints = LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/image.json"))
      .get.map(lm => lm.point)
    val correspondencePoints = modelLandmarkPoints
      .map(point => shapeModel.referenceMesh.pointSet.findClosestPoint(point).id)
      .zip(targetLandmarkPoints)
    val contourPoints = LandmarkIO.readLandmarksJson[_3D](
      new java.io.File("./datasets/validationset/annotations/liver-011.json")
    ).get.map(lm => lm.point)

    val discreteTargetImage = ImageIO.read3DScalarImageAsType[Float](new java.io.File("datasets/testset/volume-ct/liver-016.nii")).get

    val targetImage = discreteTargetImage.interpolate(BSplineImageInterpolator3D[Float](3))
    ui.show(imgGroup, discreteTargetImage, "image")

    //new
    val modelLandmarks=ui.show(modelGroup,modelLandmarkPoints.toIndexedSeq, "ModelLM")
    modelLandmarks.color=Color.BLUE


    val targetLandmarks=ui.show(imgGroup,targetLandmarkPoints.toIndexedSeq, "TargetLM")
    targetLandmarks.color=Color.RED

    //val asm=activeShapeModel

   // val modelGroup1 = ui.createGroup("modelGroup1")
   // val modelView = ui.show(modelGroup1, asm.statisticalModel, "shapeModel")

   fitMesh(activeShapeModel, correspondencePoints, contourPoints, discreteTargetImage, false)
   // val filename_landmark=new File("Data/liver_test_landmarks.json")
    //LandmarkIO.writeLandmarksJson(modelLandmarkPoints.toIndexedSeq, filename_landmark)
   //val modelLandmarks1=ui.show(modelGroup,modelLandmarkPoints.toIndexedSeq, "ModelLM")
    //modelLandmarks1.color=Color.GREEN
    //fitMesh1(activeShapeModel, correspondencePoints, contourPoints, discreteTargetImage, false)



    //val asm=activeShapeModel

    //val modelGroup1 = ui.createGroup("modelGroup1")
   // val modelView = ui.show(modelGroup1, asm.statisticalModel, "shapeModel")



  }


}
