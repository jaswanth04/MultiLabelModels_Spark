package MultiLabel

import Individual.{IndividualModel, IndividualModelBuilder}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col

abstract class MultiLabelModelBuilder[M ,L ,+IL <: IndividualModelBuilder[M, L], IM <: IndividualModel[M]]{

  def buildIndividualLearners(trainDf: DataFrame,
                              testDf: DataFrame,
                              responseColName: String,
                              vectorizedFeatureColName: String): IL
  def featureColumnNames: Seq[String]
  def responseColumnNames: Seq[String]
  def vectorizedFeatureColName: String
  val learnerName: String
  val trainData: DataFrame
  val testData: DataFrame

  def vectorizeDf(df: DataFrame, featureCols: Seq[String], vectorizedCol: String): DataFrame = {
    lazy val vecAss: VectorAssembler = new VectorAssembler().setInputCols(featureCols.toArray).setOutputCol(vectorizedCol)
    vecAss.transform(df)
  }

  def createFeatureVectorDF(df: DataFrame, requiredColumnNames: Seq[String]): DataFrame = {
    println("Combing features into a vector column:  " + vectorizedFeatureColName)
    val modifiedDf = vectorizeDf(df, this.featureColumnNames, this.vectorizedFeatureColName)
    val reqCols = requiredColumnNames ++ Seq(vectorizedFeatureColName) ++ responseColumnNames
    modifiedDf.select(reqCols.map(col): _*)
  }

  def buildBinaryRelevanceModels(trainDf: DataFrame,
                                 testDf: DataFrame,
                                 responseColumns: Seq[String],
                                 modelList: List[IndividualModel[M]]): List[IndividualModel[M]] = responseColumns match {
    case Seq() => modelList
    case head :: tail =>
      val learner: IL = buildIndividualLearners(trainDf, testDf, head, this.vectorizedFeatureColName)
      val model: IndividualModel[M] = learner.buildModel
      buildBinaryRelevanceModels(trainDf = model.trainPredictions,
        testDf = model.testPredictions,
        responseColumns = tail,
        modelList = modelList ++ List(model))


  }

  def buildClassifierChainModels(trainDf: DataFrame,
                                 testDf: DataFrame,
                                 responseColumns: Seq[String],
                                 vectorizedFeatureName: String,
                                 modelList: List[IndividualModel[M]]): List[IndividualModel[M]] = responseColumns match {
    case Seq() => modelList
    case head :: tail =>
      val learner: IL = buildIndividualLearners(trainDf, testDf, head, vectorizedFeatureColName)
      val model: IndividualModel[M] = learner.buildModel
      val vecAss = new VectorAssembler()
        .setInputCols(Array(vectorizedFeatureName, head ++ "_prediction"))
        .setOutputCol("feature_" ++ head)
      buildClassifierChainModels(trainDf = vecAss.transform(model.trainPredictions).drop(vectorizedFeatureName).persist(),
        testDf = vecAss.transform(model.testPredictions).drop(vectorizedFeatureName).persist(),
        responseColumns = tail,
        vectorizedFeatureName = "feature_" ++ head,
        modelList = modelList ++ List(model))

  }

  def buildMultiLabelModels(multiLabelAlgorithm: String): MultiLabelModel[M, IM]
}
