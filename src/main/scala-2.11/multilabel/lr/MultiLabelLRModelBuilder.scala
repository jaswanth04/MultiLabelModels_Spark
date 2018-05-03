package multilabel.lr

import individual.lr.{IndividualLRModel, IndividualLRModelBuilder}
import multilabel.MultiLabelModelBuilder
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.sql.DataFrame

class MultiLabelLRModelBuilder(data: DataFrame,
                               xFlagNames: Seq[String],
                               featureColName: String = "features",
                               responseColumns: Seq[String],
                               reqColsInPrediction: Seq[String] = Seq(),
                               maxIterations: Int = 100,
                               regularizationParameter: Double = 0.0,
                               elasticNetParameter: Double = 0.0,
                               trainTestSplitFunc: DataFrame => (DataFrame, DataFrame))
extends MultiLabelModelBuilder[LogisticRegressionModel,
  LogisticRegression,
  IndividualLRModelBuilder,
  IndividualLRModel] {

  val featureColumnNames: Seq[String] = xFlagNames
  val responseColumnNames: Seq[String] = responseColumns
  val vectorizedFeatureColName: String = featureColName
  val learnerName = "LogisticRegression"
  private val vectorizedDf = createFeatureVectorDF(data, reqColsInPrediction)
  vectorizedDf.cache()
  val (trainData, testData) = trainTestSplitFunc(vectorizedDf)

  def buildIndividualLearners(trainDf: DataFrame,
                              responseColName: String,
                              featureColumnName: String): IndividualLRModelBuilder = new IndividualLRModelBuilder(trainDf = trainDf,
    testDf = this.testData,
    responseColName = responseColName,
    vectorizedFeatureColName = featureColumnName,
    reqColsInPredictions = reqColsInPrediction,
    maxIterations = maxIterations,
    regularizationParameter = regularizationParameter,
    elasticNetParameter = elasticNetParameter)

  def buildMultiLabelModels(multiLabelAlgorithm: String): MultiLabelLRModel = {
    if (multiLabelAlgorithm == "BR") {
      val brModels = buildBinaryRelevanceModels(trainDf = trainData,
        responseColumns = responseColumnNames)
      new MultiLabelLRModel(modelList = brModels.map(_.asInstanceOf[IndividualLRModel]),
        modelType = "BR",
      trainDf = trainData,
      testDf = testData,
      reqCols = reqColsInPrediction)}
    else {
      val ccModels = buildClassifierChainModels(trainDf = trainData,
        responseColumns = responseColumnNames,
        vectorizedFeatureName = vectorizedFeatureColName)
      new MultiLabelLRModel(modelList = ccModels.map(_.asInstanceOf[IndividualLRModel]),
        modelType = "CC",
      trainDf = trainData,
      testDf = testData,
      reqCols = reqColsInPrediction)}
  }


}
