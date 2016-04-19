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

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.param.ParamsSuite
import org.apache.spark.ml.util.DefaultReadWriteTest
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.mllib.util.TestingUtils._
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

class RankingEvaluatorSuite
  extends SparkFunSuite with MLlibTestSparkContext with DefaultReadWriteTest {

  test("params") {
    ParamsSuite.checkParams(new RankingEvaluator)
  }

  test("ranking evaluation metrics") {
    // test IntegerType for query col and StringType for actual and prediction cols
    val schema = StructType(
      Seq(
        StructField( "query", IntegerType, false),
        StructField( "label", StringType, true),
        StructField( "prediction", StringType, true),
        StructField( "relevance", DoubleType, true)
      )
    )
    val data = sc.parallelize(Seq(
      // query 1 has 2 documents in ground truth
      // 2nd predicted position is relevant
      (1, "10", "10", 0.9),   // doc in actual and predicted
      (1, "20", null, null),  // doc in actual but not in predicted
      (1, null, "40", 1.0),   // doc in predicted but not in actual
      // query 2 has 3 items in ground truth
      // 1st predicted position is relevant
      (2, "40", null, null),
      (2, "30", "30", 1.0),
      (2, "10", null, null),
      (2, null, "20", 0.8),
      // query 3 has 2 items in ground truth
      // no predictions are relevant
      (3, "20", null, null),
      (3, "30", null, null),
      (3, null, "10", 0.7),
      (3, null, "40", 0.5),
      // query 4 has 3 items in ground truth
      // all predictions are relevant
      (4, "10", "10", 0.6),
      (4, "20", "20", 0.9),
      (4, "30", null, null),
      (4, "40", null, null)
    )).map { case (u, a, p, r) => Row(u, a, p, r) }
    val dataset = sqlContext.createDataFrame(data, schema)

    // map (default)
    val evaluator = new RankingEvaluator()
    assert(evaluator.evaluate(dataset) ~== 0.2708 absTol 1e-2)
    // mapk
    evaluator.setK(2)
    evaluator.setMetricName("mapk")
    assert(evaluator.evaluate(dataset) ~== 0.5 absTol 1e-2)
    // ndcg
    evaluator.setMetricName("ndcg")
    assert(evaluator.evaluate(dataset) ~== 0.5 absTol 1e-2)
  }

  test("RankingEvaluator requires same type for label and prediction col") {
    val local = sqlContext
    import local.implicits._

    val df = Seq(
      (1, 1, "1", 1.0),
      (1, 2, "2", 0.9)
    ).toDF("query", "label", "prediction", "relevance")
    intercept[IllegalArgumentException] {
      new RankingEvaluator().evaluate(df)
    }
  }

  test("read/write") {
    val evaluator = new RankingEvaluator()
      .setQueryCol("user")
      .setLabelCol("ground_truth_items")
      .setPredictionCol("recommended_items")
      .setRelevanceCol("score")
      .setK(5)
      .setMetricName("ndcg")
    testDefaultReadWrite(evaluator)
  }
}
