package individual.lsvc

import individual.IndividualModelBuilder
import org.apache.spark.ml.classification.{LinearSVC, LinearSVCModel}
import org.apache.spark.sql.DataFrame

class IndividualLSVCModelBuilder(trainDf: DataFrame,
                                 vectorizedFeatureColName: String,
                                 responseColName: String,
                                 reqColsInPredictions: Seq[String] = Seq(),
                                 maxIterations: Int = 100,
                                 regularizationParameter: Double = 0.0,
                                 standardization: Boolean = true,
                                 tolerance: Double = 1e-6,
                                 testDf:DataFrame) extends
  IndividualModelBuilder[LinearSVCModel, LinearSVC](featureColName = vectorizedFeatureColName,
    trainDf = trainDf,
    testDf = testDf,
    responseColName = responseColName) {

  val learnerName = "LinearSupportVectorClassification"
  override val reqCols: Seq[String] = reqColsInPredictions

  println(s"Building model for the feature: $vectorizedFeatureColName")

  def learner: LinearSVC = new LinearSVC()
    .setFeaturesCol(vectorizedFeatureColName)
    .setLabelCol(responseColumnName)
    .setMaxIter(maxIterations)
    .setRegParam(regularizationParameter)
    .setStandardization(standardization)
    .setTol(1e-9)
    .setPredictionCol(predictionCols.head)
    .setRawPredictionCol(predictionCols(2))

  def buildModel: IndividualLSVCModel = new IndividualLSVCModel(lsvcModel = learner.fit(trainData),
    testData = testData,
    trainData = trainData,
    reqColsInPrediction = this.reqColsInPredictions)


}
