package Individual.LR

import Individual.IndividualModelBuilder
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.sql.DataFrame

class IndividualLRModelBuilder(trainDf: DataFrame,
                               vectorizedFeatureColName: String,
                               responseColName: String,
                               reqColsInPredictions: Seq[String] = Seq(),
                               maxIterations: Int = 100,
                               regularizationParameter: Double = 0.0,
                               elasticNetParameter: Double = 0.0,
                               testDf:DataFrame) extends
  IndividualModelBuilder[LogisticRegressionModel, LogisticRegression](featureColName = vectorizedFeatureColName,
    trainDf = trainDf,
    testDf = testDf,
    responseColName = responseColName) {

  val learnerName = "LogisticRegression"
  override val reqCols: Seq[String] = reqColsInPredictions

  println(s"Building model for the feature: $vectorizedFeatureColumnName")

  type L = LogisticRegression

  def learner: LogisticRegression = new LogisticRegression()
    .setFeaturesCol(vectorizedFeatureColumnName)
    .setLabelCol(responseColumnName)
    .setMaxIter(maxIterations)
    .setRegParam(regularizationParameter)
    .setElasticNetParam(elasticNetParameter)
    .setTol(1e-9)
    .setPredictionCol(predictionCols.head)
    .setProbabilityCol(predictionCols(1))
    .setRawPredictionCol(predictionCols(2))

  def buildModel: IndividualLRModel = new IndividualLRModel(lrModel = learner.fit(trainData),
    testData = testData,
    trainData = trainData,
    reqColsInPrediction = this.reqColsInPredictions)

}
