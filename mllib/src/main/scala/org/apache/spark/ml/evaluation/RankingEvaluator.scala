/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.evaluation

import org.apache.spark.annotation.{Experimental, Since}
import org.apache.spark.ml.evaluation.RankingEvaluator.RankingAggregator
import org.apache.spark.ml.param.{IntParam, Param, ParamMap, ParamValidators}
import org.apache.spark.ml.param.shared.{HasLabelCol, HasPredictionCol}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable, SchemaUtils}
import org.apache.spark.mllib.evaluation.RankingMetrics
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

/**
 * :: Experimental ::
 * Evaluator for ranking, which expects four input columns: query, label, prediction and relevance.
 * The 'query' column determines the 'query id' by which actual and predicted values are grouped to
 * generate the set of ground truth and predicted documents, respectively.
 * The 'label' column contains the actual 'document ids' (can contain nulls).
 * The 'prediction' column containts the predicted 'document ids', while the 'relevance' column
 * contains the predicted relevance scores for each predicted 'document' (can contain nulls).
 */
@Since("2.0.0")
@Experimental
final class RankingEvaluator @Since("2.0.0") (@Since("2.0.0") override val uid: String)
  extends Evaluator with HasPredictionCol with HasLabelCol with DefaultParamsWritable {

  @Since("2.0.0")
  def this() = this(Identifiable.randomUID("rankEval"))

  /**
   * Param for metric name in evaluation. Supports:
   *  - `"map"`: mean average precision (default)
   *  - `"mapk"`: mean average precision @ k
   *  - `"ndcg"`: normalized discounted cumulative gain @ k
   *
   * @group param
   */
  @Since("2.0.0")
  val metricName: Param[String] = {
    val isValidMetric = ParamValidators.inArray(RankingEvaluator.supportedMetrics)
    new Param(this, "metricName", s"metric name in evaluation. Supported values: " +
      s"${RankingEvaluator.supportedMetrics.mkString(",")}", isValidMetric)
  }

  /** @group getParam */
  @Since("2.0.0")
  def getMetricName: String = $(metricName)

  /** @group setParam */
  @Since("2.0.0")
  def setMetricName(value: String): this.type = set(metricName, value)

  /**
   * Param for threshold for predicted set (applicable for metrics 'ndcg' and 'mapk', ignored
   * otherwise).
   *
   * @group param
   */
  val k = new IntParam(this, "k", "threshold for predicted set (only applicable for metrics " +
    "'ndcg' and 'mapk', ignored otherwise).", ParamValidators.gt(0))

  /** @group getParam */
  @Since("2.0.0")
  def getK: Int = $(k)

  /** @group setParam */
  @Since("2.0.0")
  def setK(value: Int): this.type = set(k, value)

  /**
   * Param for column name containing the 'query id'. If set, the evaluator will group the input
   * dataset by this column, generating a set of ground truth 'document ids' (taken from
   * [[labelCol]]) and 'predicted ids' (taken from [[predictionCol]]) for each unique id in
   * 'queryCol'.
   *
   * @group param
   */
  val queryCol: Param[String] = new Param(this, "queryCol", "column name containing the " +
    "'query id'. If set, the evaluator will group the input dataset by this column, generating" +
    " a set of ground truth 'document ids' (taken from 'labelCol') and 'predicted ids' " +
    "(taken from 'predictionCol') for each unique id in 'queryCol'.")

  /** @group getParam */
  @Since("2.0.0")
  def getQueryCol: String = $(queryCol)

  /** @group setParam */
  @Since("2.0.0")
  def setQueryCol(value: String): this.type = set(queryCol, value)

  /**
   * Param for column name containing the 'relevance score' for each predicted document
   * in [[predictionCol]].
   *
   * @group param
   */
  val relevanceCol: Param[String] = new Param(this, "relevanceCol", "column name containing the " +
    "'relevance score for each predicted document in 'predictionCol'.")

  /** @group getParam */
  @Since("2.0.0")
  def getRelevanceCol: String = $(relevanceCol)

  /** @group setParam */
  @Since("2.0.0")
  def setRelevanceCol(value: String): this.type = set(relevanceCol, value)

  /** @group setParam */
  @Since("2.0.0")
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /** @group setParam */
  @Since("2.0.0")
  def setLabelCol(value: String): this.type = set(labelCol, value)

  setDefault(metricName -> "map", k -> 10, queryCol -> "query", relevanceCol -> "relevance")

  private def validateSchema(dataset: Dataset[_]): Unit = {
    val schema = dataset.schema
    require(schema($(labelCol)).dataType.sameType(schema($(predictionCol)).dataType))
    // we handle common types for 'query id', 'document id'
    // val idTypes = Seq(IntegerType, LongType, StringType)
    // SchemaUtils.checkColumnTypes(schema, $(predictionCol), idTypes)
    // SchemaUtils.checkColumnTypes(schema, $(labelCol), idTypes)
    // relevance defines the ordering of predicted values, can be any numeric type
    SchemaUtils.checkNumericType(schema, $(relevanceCol))
  }

  @Since("2.0.0")
  override def evaluate(dataset: Dataset[_]): Double = {
    validateSchema(dataset)
    val predictedAndActual =
      dataset.select(col($(queryCol)), col($(labelCol)), col($(predictionCol)),
        col($(relevanceCol)).cast(DoubleType))
      .rdd
      .map { case row =>
        val qid = row.get(0)
        // label, prediction, and relevance cols are nullable, so wrap them in Options
        val actual = Option(row.get(1))
        val predicted = Option(row.get(2)).map((_, row.getDouble(3)))
        (qid, (predicted, actual))
      }.aggregateByKey(new RankingAggregator)(
        // TODO optimize for known k?
        // TODO use native Spark versions of 'collect_list' and 'collect_set' when available?
        seqOp = { case (agg, (predScores, actual)) => agg.add(actual, predScores) },
        combOp = { case (left, right) => new RankingAggregator().merge(left).merge(right) }
      ).mapValues(agg => (agg.getPredicted, agg.getActuals)).values

    val metrics = new RankingMetrics(predictedAndActual)
    $(metricName) match {
      case "map" => metrics.meanAveragePrecision
      case "mapk" => metrics.precisionAt($(k))
      case "ndcg" => metrics.ndcgAt($(k))
      // TODO case "mrr" => metrics.mrr
    }
  }

  @Since("1.5.0")
  override def copy(extra: ParamMap): RankingEvaluator = defaultCopy(extra)
}

@Since("2.0.0")
object RankingEvaluator extends DefaultParamsReadable[RankingEvaluator] {

  private final val supportedMetrics = Array("map", "mapk", "ndcg")

  private class RankingAggregator extends Serializable {
    import scala.collection.mutable

    val ord: Ordering[(Any, Double)] = Ordering.by(-_._2)
    private val actuals = mutable.Set[Any]()
    private val predictedWithScores = mutable.SortedSet[(Any, Double)]()(ord)

    /** Add an optional actual value, and an optional (predicted, score) pair to aggregator */
    def add(actual: Option[Any], predicted: Option[(Any, Double)]): this.type = {
      actual.foreach(actuals += _)
      predicted.foreach(predictedWithScores += _)
      this
    }

    def merge(other: RankingAggregator): this.type = {
      actuals ++= other.actuals
      predictedWithScores ++= other.predictedWithScores
      this
    }

    def getActuals: Array[Any] = actuals.toArray

    def getPredicted: Array[Any] = predictedWithScores.toArray.map(_._1)
  }

  @Since("1.6.0")
  override def load(path: String): RankingEvaluator = super.load(path)
}