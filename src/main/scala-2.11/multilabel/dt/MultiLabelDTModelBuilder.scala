package multilabel.dt

import individual.dt.{IndividualDTModel, IndividualDTModelBuilder}
import multilabel.MultiLabelModelBuilder
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.sql.DataFrame

class MultiLabelDTModelBuilder(data: DataFrame,
                               xFlagNames: Seq[String],
                               featureColName: String = "features",
                               responseColumns: Seq[String],
                               reqColsInPrediction: Seq[String] = Seq(),
                               impurity: String = "gini",
                               maxBins: Int = 32,
                               maxDepth: Int = 5,
                               minInfoGain: Double = 0.0,
                               minInstancesPerNode : Int = 1,
                               trainTestSplitFunc: DataFrame => (DataFrame, DataFrame))
  extends MultiLabelModelBuilder[DecisionTreeClassificationModel,
    DecisionTreeClassifier,
    IndividualDTModelBuilder,
    IndividualDTModel] {

  val featureColumnNames: Seq[String] = xFlagNames
  val responseColumnNames: Seq[String] = responseColumns
  val vectorizedFeatureColName: String = featureColName
  val learnerName = "LogisticRegression"
  private val vectorizedDf = createFeatureVectorDF(data, reqColsInPrediction)
  vectorizedDf.cache()
  val (trainData, testData) = trainTestSplitFunc(vectorizedDf)

  def buildIndividualLearners(trainDf: DataFrame,
                              responseColName: String,
                              featureColumnName: String): IndividualDTModelBuilder = new IndividualDTModelBuilder(trainDf = trainDf,
    testDf = this.testData,
    responseColName = responseColName,
    vectorizedFeatureColName = featureColumnName,
    reqColsInPredictions = reqColsInPrediction,
    impurity = impurity,
    maxBins = maxBins,
    maxDepth = maxDepth,
    minInfoGain = minInfoGain,
    minInstancesPerNode = minInstancesPerNode)

  def buildMultiLabelModels(multiLabelAlgorithm: String): MultiLabelDTModel = {
    if (multiLabelAlgorithm == "BR") {
      val brModels = buildBinaryRelevanceModels(trainDf = trainData,
        responseColumns = responseColumnNames)
      new MultiLabelDTModel(modelList = brModels.map(_.asInstanceOf[IndividualDTModel]),
        modelType = "BR",
        trainDf = trainData,
        testDf = testData,
        reqCols = reqColsInPrediction)}
    else {
      val ccModels = buildClassifierChainModels(trainDf = trainData,
        responseColumns = responseColumnNames,
        vectorizedFeatureName = vectorizedFeatureColName)
      new MultiLabelDTModel(modelList = ccModels.map(_.asInstanceOf[IndividualDTModel]),
        modelType = "CC",
        trainDf = trainData,
        testDf = testData,
        reqCols = reqColsInPrediction)}
  }

}
