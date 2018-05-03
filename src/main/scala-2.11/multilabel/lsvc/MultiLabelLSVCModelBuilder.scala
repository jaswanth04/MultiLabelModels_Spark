package multilabel.lsvc

import individual.lsvc.{IndividualLSVCModel, IndividualLSVCModelBuilder}
import multilabel.MultiLabelModelBuilder
import org.apache.spark.ml.classification.{LinearSVC, LinearSVCModel}
import org.apache.spark.sql.DataFrame

class MultiLabelLSVCModelBuilder(data: DataFrame,
                                 xFlagNames: Seq[String],
                                 featureColName: String = "features",
                                 responseColumns: Seq[String],
                                 reqColsInPrediction: Seq[String] = Seq(),
                                 maxIterations: Int = 100,
                                 regularizationParameter: Double = 0.0,
                                 standardization: Boolean = true,
                                 tolerance: Double = 1e-6,
                                 trainTestSplitFunc: DataFrame => (DataFrame, DataFrame))
  extends MultiLabelModelBuilder[LinearSVCModel,
    LinearSVC,
    IndividualLSVCModelBuilder,
    IndividualLSVCModel] {

  val learnerName = "LinearSupportVectorClassifier"

  val featureColumnNames: Seq[String] = xFlagNames
  val responseColumnNames: Seq[String] = responseColumns
  val vectorizedFeatureColName: String = featureColName

  private val vectorizedDf = createFeatureVectorDF(data, reqColsInPrediction)
  vectorizedDf.cache()
  val (trainData, testData) = trainTestSplitFunc(vectorizedDf)

  def buildIndividualLearners(trainDf: DataFrame,
                              responseColName: String,
                              featureColumnName: String): IndividualLSVCModelBuilder = new IndividualLSVCModelBuilder(trainDf = trainDf,
    testDf = this.testData,
    responseColName = responseColName,
    vectorizedFeatureColName = featureColumnName,
    reqColsInPredictions = reqColsInPrediction,
    maxIterations = maxIterations,
    regularizationParameter = regularizationParameter,
    standardization = standardization,
    tolerance = tolerance)

  def buildMultiLabelModels(multiLabelAlgorithm: String): MultiLabelLSVCModel = {
    if (multiLabelAlgorithm == "BR") {
      val brModels = buildBinaryRelevanceModels(trainDf = trainData,
        responseColumns = responseColumnNames)
      new MultiLabelLSVCModel(modelList = brModels.map(_.asInstanceOf[IndividualLSVCModel]),
        modelType = "BR",
        trainDf = trainData,
        testDf = testData,
        reqCols = reqColsInPrediction)}
    else {
      val ccModels = buildClassifierChainModels(trainDf = trainData,
        responseColumns = responseColumnNames,
        vectorizedFeatureName = vectorizedFeatureColName)
      new MultiLabelLSVCModel(modelList = ccModels.map(_.asInstanceOf[IndividualLSVCModel]),
        modelType = "CC",
        trainDf = trainData,
        testDf = testData,
        reqCols = reqColsInPrediction)}
  }

}
