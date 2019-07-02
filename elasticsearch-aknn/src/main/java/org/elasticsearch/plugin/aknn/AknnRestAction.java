/*
 * Copyright [2018] [Alex Klibisz]
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package org.elasticsearch.plugin.aknn;

import org.elasticsearch.action.bulk.BulkRequestBuilder;
import org.elasticsearch.action.bulk.BulkResponse;
import org.elasticsearch.action.get.GetResponse;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.client.node.NodeClient;
import org.elasticsearch.common.StopWatch;
import org.elasticsearch.common.inject.Inject;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentHelper;
import org.elasticsearch.common.xcontent.XContentParser;
import org.elasticsearch.common.xcontent.XContentFactory;
import org.elasticsearch.common.xcontent.DeprecationHandler;
import org.elasticsearch.common.Strings;
import org.elasticsearch.index.query.BoolQueryBuilder;
import org.elasticsearch.index.query.QueryBuilders;
import org.elasticsearch.index.query.WrapperQueryBuilder;
import org.elasticsearch.rest.BaseRestHandler;
import org.elasticsearch.rest.BytesRestResponse;
import org.elasticsearch.rest.RestController;
import org.elasticsearch.rest.RestRequest;
import org.elasticsearch.rest.RestStatus;
import org.elasticsearch.search.SearchHit;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import static java.lang.Math.*;
import static org.elasticsearch.rest.RestRequest.Method.GET;
import static org.elasticsearch.rest.RestRequest.Method.POST;

@SuppressWarnings("FieldCanBeLocal")
public class AknnRestAction extends BaseRestHandler {

    private static String NAME = "_aknn";
    private final String NAME_SEARCH = "_aknn_search";
    private final String NAME_SEARCH_VEC = "_aknn_search_vec";
    private final String NAME_INDEX = "_aknn_index";
    private final String NAME_CREATE = "_aknn_create";
    private final String NAME_CLEAR_CACHE = "_aknn_clear_cache";

    // TODO: check how parameters should be defined at the plugin level.
    private final String HASHES_KEY = "_aknn_hashes";
    private final String VECTOR_KEY = "_aknn_vector";
    private final int K1_DEFAULT = 99;
    private final int K2_DEFAULT = 10;
    private final String DISTANCE_DEFAULT = "euclidean";
    private final boolean RESCORE_DEFAULT = true;
    private final int MINIMUM_DEFAULT = 1;

    // TODO: add an option to the index endpoint handler that empties the cache.
    private Map<String, LshModel> lshModelCache = new HashMap<>();

    @Inject
    AknnRestAction(Settings settings, RestController controller) {
        super(settings);
        controller.registerHandler(GET, "/{index}/{type}/{id}/" + NAME_SEARCH, this);
        controller.registerHandler(POST, NAME_SEARCH_VEC, this);
        controller.registerHandler(POST, NAME_INDEX, this);
        controller.registerHandler(POST, NAME_CREATE, this);
        controller.registerHandler(GET, NAME_CLEAR_CACHE, this);
    }

    // @Override
    public String getName() {
        return NAME;
    }

    @Override
    protected RestChannelConsumer prepareRequest(RestRequest restRequest, NodeClient client) throws IOException {
        if (restRequest.path().endsWith(NAME_SEARCH_VEC))
            return handleSearchVecRequest(restRequest, client);
        else if (restRequest.path().endsWith(NAME_SEARCH))
            return handleSearchRequest(restRequest, client);
        else if (restRequest.path().endsWith(NAME_INDEX))
            return handleIndexRequest(restRequest, client);
        else if (restRequest.path().endsWith(NAME_CLEAR_CACHE))
            return handleClearRequest(restRequest, client);
        else
            return handleCreateRequest(restRequest, client);
    }

    private Double calculateScore(List<Double> queryVector, List<Double> hitVector, String distance) {

        double score;

        switch(distance) {
            case "cosine": score = cosineDistance(queryVector, hitVector);
                break;
            case "euclidean": score = euclideanDistance(queryVector, hitVector);
                break;
            default: throw new RuntimeException("Invalid distance type: " + distance);
        }

        return score;
    }

    private static Double euclideanDistance(List<Double> A, List<Double> B) {

        double squaredDistance = 0.0;

        for (int i = 0; i < A.size(); i++) {
            squaredDistance += Math.pow(A.get(i) - B.get(i), 2);
        }

        return sqrt(squaredDistance);
    }

    private static Double cosineDistance(List<Double> A, List<Double> B) {

        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < A.size(); i++) {
            dotProduct += A.get(i) * B.get(i);
            normA += Math.pow(A.get(i), 2);
            normB += Math.pow(B.get(i), 2);
        }

        // Avoid negative numbers in case the cosine is > 1.0 (it could happen in case of computational approximation).
        return max(0.0, 1.0 - dotProduct / (sqrt(normA) * sqrt(normB)));
    }

    // Loading LSH model refactored as function
    // TODO Fix issues with stopwatch
    private LshModel initLsh(String aknnURI, NodeClient client) {
        LshModel lshModel;
        StopWatch stopWatch = new StopWatch("StopWatch to load LSH cache");
        if (!lshModelCache.containsKey(aknnURI)) {

            // Get the Aknn document.
            logger.info("Get Aknn model document from {}", aknnURI);
            stopWatch.start("Get Aknn model document");
            String[] annURITokens = aknnURI.split("/");
            GetResponse aknnGetResponse = client.prepareGet(annURITokens[0], annURITokens[1], annURITokens[2]).get();
            stopWatch.stop();

            // Instantiate LSH from the source map.
            logger.info("Parse Aknn model document");
            stopWatch.start("Parse Aknn model document");
            lshModel = LshModel.fromMap(aknnGetResponse.getSourceAsMap());
            stopWatch.stop();

            // Save for later.
            lshModelCache.put(aknnURI, lshModel);

        } else {
            logger.info("Get Aknn model document from local cache");
            stopWatch.start("Get Aknn model document from local cache");
            lshModel = lshModelCache.get(aknnURI);
            stopWatch.stop();
        }
        return lshModel;
    }

    //  Query execution refactored as function and added wrapper query
    private List<Map<String, Object>> queryLsh(List<Double> queryVector,
                                               Map<String, Long> queryHashes,
                                               String index,
                                               String type,
                                               Integer k1,
                                               Boolean rescore,
                                               String distance,
                                               String filterString,
                                               Integer minimum_should_match,
                                               Boolean debug,
                                               NodeClient client) {

        // Retrieve the documents with most matching hashes. https://stackoverflow.com/questions/10773581
        StopWatch stopWatch = new StopWatch("StopWatch to query LSH cache");
        logger.info("Build boolean query from hashes");
        stopWatch.start("Build boolean query from hashes");
        BoolQueryBuilder queryBuilder = QueryBuilders.boolQuery();
        for (Map.Entry<String, Long> entry : queryHashes.entrySet()) {
            String termKey = HASHES_KEY + "." + entry.getKey();
            queryBuilder.should(QueryBuilders.termQuery(termKey, entry.getValue()));
        }
        queryBuilder.minimumShouldMatch(minimum_should_match);

        if (filterString != null) {
            queryBuilder.filter(new WrapperQueryBuilder(filterString));
        }
        //logger.info(queryBuilder.toString());
        stopWatch.stop();

        String hashes = debug ? null : HASHES_KEY;

        logger.info("Execute boolean search");
        stopWatch.start("Execute boolean search");
        SearchResponse approximateSearchResponse = client
                .prepareSearch(index)
                .setTypes(type)
                .setFetchSource("*", hashes)
                .setQuery(queryBuilder)
                .setSize(k1)
                .get();
        stopWatch.stop();

        // Compute exact KNN on the approximate neighbors.
        // Recreate the SearchHit structure, but remove the vector and hashes.
        logger.info("Compute exact distance and construct search hits. Distance function: " + distance);
        stopWatch.start("Compute exact distance and construct search hits");

        List<Map<String, Object>> modifiedSortedHits = new ArrayList<>();

        for (SearchHit hit : approximateSearchResponse.getHits()) {
            Map<String, Object> hitSource = hit.getSourceAsMap();
            @SuppressWarnings("unchecked")
            List<Double> hitVector = (List<Double>) hitSource.get(VECTOR_KEY);
            if (!debug) {
                hitSource.remove(VECTOR_KEY);
                hitSource.remove(HASHES_KEY);
            }
            if (rescore) {
                modifiedSortedHits.add(new HashMap<String, Object>() {{
                    put("_index", hit.getIndex());
                    put("_id", hit.getId());
                    put("_type", hit.getType());
                    put("_score", calculateScore(queryVector, hitVector, distance));
                    put("_source", hitSource);
                }});
            } else {
                modifiedSortedHits.add(new HashMap<String, Object>() {{
                    put("_index", hit.getIndex());
                    put("_id", hit.getId());
                    put("_type", hit.getType());
                    put("_score", hit.getScore());
                    put("_source", hitSource);
                }});
            }
        }
        stopWatch.stop();

        if (rescore) {
            logger.info("Sort search hits by exact distance");
            stopWatch.start("Sort search hits by exact distance");
            modifiedSortedHits.sort(Comparator.comparingDouble(x -> (Double) x.get("_score")));
            stopWatch.stop();
        } else {
            logger.info("Exact distance rescoring passed");
        }

        logger.info("Timing summary for querying\n {}", stopWatch.prettyPrint());

        return modifiedSortedHits;
    }

    private RestChannelConsumer handleSearchRequest(RestRequest restRequest, NodeClient client) {
        /*
         * Original handleSearchRequest() refactored for further reusability
         * and added some additional parameters, such as filter query.
         *
         * @param  index    Index name
         * @param  type     Doc type (keep in mind forthcoming _type removal in ES7)
         * @param  id       Query document id
         * @param  filter   String in format of ES bool query filter (excluding
         *                  parent 'filter' node)
         * @param  k1       Number of candidates for scoring
         * @param  k2       Number of hits returned
         * @param  distance The type of the algorithm to use to calculate the vectors distance
         * @param  minimum_should_match    number of hashes should match for hit to be returned
         * @param  rescore  If set to 'True' will return results without exact matching stage
         * @param  debug    If set to 'True' will include original vectors and hashes in hits
         */

        StopWatch stopWatch = new StopWatch("StopWatch to Time Search Request");

        // Parse request parameters.
        stopWatch.start("Parse request parameters");
        final String index = restRequest.param("index");
        final String type = restRequest.param("type");
        final String id = restRequest.param("id");
        final String filter = restRequest.param("filter", null);
        final int k1 = restRequest.paramAsInt("k1", K1_DEFAULT);
        final int k2 = restRequest.paramAsInt("k2", K2_DEFAULT);
        final String distance = restRequest.param("distance", DISTANCE_DEFAULT);
        final int minimum_should_match = restRequest.paramAsInt("minimum_should_match", MINIMUM_DEFAULT);
        final boolean rescore = restRequest.paramAsBoolean("rescore", RESCORE_DEFAULT);
        final boolean debug = restRequest.paramAsBoolean("debug", false);
        stopWatch.stop();

        logger.info("Get query document at {}/{}/{}", index, type, id);
        stopWatch.start("Get query document");
        GetResponse queryGetResponse = client.prepareGet(index, type, id).get();
        Map<String, Object> baseSource = queryGetResponse.getSource();
        stopWatch.stop();

        logger.info("Parse query document hashes");
        stopWatch.start("Parse query document hashes");
        @SuppressWarnings("unchecked")
        Map<String, Long> queryHashes = (Map<String, Long>) baseSource.get(HASHES_KEY);
        stopWatch.stop();

        stopWatch.start("Parse query document vector");
        @SuppressWarnings("unchecked")
        List<Double> queryVector = (List<Double>) baseSource.get(VECTOR_KEY);
        stopWatch.stop();

        stopWatch.start("Query nearest neighbors");
        List<Map<String, Object>> modifiedSortedHits =
                queryLsh(queryVector, queryHashes, index, type, k1, rescore, distance, filter, minimum_should_match,
                        debug, client);

        stopWatch.stop();

        logger.info("Timing summary\n {}", stopWatch.prettyPrint());

        return channel -> {
            XContentBuilder builder = channel.newBuilder();
            builder.startObject();
            builder.field("took", stopWatch.totalTime().getMillis());
            builder.field("timed_out", false);
            builder.startObject("hits");
            builder.field("max_score", 0);

            // In some cases there will not be enough approximate matches to return *k2* hits. For example, this could
            // be the case if the number of bits per table in the LSH model is too high, over-partioning the space.
            builder.field("total", min(k2, modifiedSortedHits.size()));
            builder.field("hits", modifiedSortedHits.subList(0, min(k2, modifiedSortedHits.size())));
            builder.endObject();
            builder.endObject();
            channel.sendResponse(new BytesRestResponse(RestStatus.OK, builder));
        };
    }

    private RestChannelConsumer handleSearchVecRequest(RestRequest restRequest, NodeClient client) throws IOException {
        /*
         * Hybrid of refactored handleSearchRequest() and handleIndexRequest()
         * Takes document containing query vector, hashes it, and executing query
         * without indexing.
         *
         * @param  index        Index name
         * @param  type         Doc type (keep in mind forthcoming _type removal in ES7)
         * @param  _aknn_vector Query vector
         * @param  filter       String in format of ES bool query filter (excluding
         *                      parent 'filter' node)
         * @param  k1           Number of candidates for scoring
         * @param  k2           Number of hits returned
         * @param  distance The type of the algorithm to use to calculate the vectors distance
         * @param  minimum_should_match    number of hashes should match for hit to be returned
         * @param  rescore      If set to 'True' will return results without exact matching stage
         * @param  debug        If set to 'True' will include original vectors and hashes in hits
         * @param  clear_cache  Force update LSH model cache before executing hashing.
         */

        StopWatch stopWatch = new StopWatch("StopWatch to Time Search Request");

        // Parse request parameters.
        stopWatch.start("Parse request parameters");
        XContentParser xContentParser = XContentHelper.createParser(
                restRequest.getXContentRegistry(),
                DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                restRequest.content(),
                restRequest.getXContentType());
        Map<String, Object> contentMap = xContentParser.mapOrdered();
        @SuppressWarnings("unchecked")
        Map<String, Object> aknnQueryMap = (Map<String, Object>) contentMap.get("query_aknn");
        @SuppressWarnings("unchecked")
        Map<String, ?> filterMap = (Map<String, ?>) contentMap.get("filter");
        String filter = null;
        if (filterMap != null) {
            XContentBuilder filterBuilder = XContentFactory.jsonBuilder().map(filterMap);
            filter = Strings.toString(filterBuilder);
        }

        final String index = (String) contentMap.get("_index");
        final String type = (String) contentMap.get("_type");
        final String aknnURI = (String) contentMap.get("_aknn_uri");
        final int k1 = (int) aknnQueryMap.get("k1");
        final int k2 = (int) aknnQueryMap.get("k2");
        final int minimum_should_match = restRequest.paramAsInt("minimum_should_match", MINIMUM_DEFAULT);
        final boolean rescore = restRequest.paramAsBoolean("rescore", RESCORE_DEFAULT);
        final boolean clear_cache = restRequest.paramAsBoolean("clear_cache", false);
        final boolean debug = restRequest.paramAsBoolean("debug", false);
        final String distance = restRequest.param("distance", DISTANCE_DEFAULT);

        @SuppressWarnings("unchecked")
        List<Double> queryVector = (List<Double>) aknnQueryMap.get(VECTOR_KEY);
        stopWatch.stop();
        // Clear LSH model cache if requested
        if (clear_cache) {
            // Clear LSH model cache
            lshModelCache.remove(aknnURI);
        }
        // Check if the LshModel has been cached. If not, retrieve the Aknn document and use it to populate the model.
        LshModel lshModel = initLsh(aknnURI, client);

        stopWatch.start("Query nearest neighbors");
        Map<String, Long> queryHashes = lshModel.getVectorHashes(queryVector);
        //logger.info("HASHES: {}", queryHashes);

        List<Map<String, Object>> modifiedSortedHits = queryLsh(queryVector, queryHashes, index, type, k1, rescore,
                distance, filter, minimum_should_match, debug, client);

        stopWatch.stop();
        logger.info("Timing summary\n {}", stopWatch.prettyPrint());
        return channel -> {
            XContentBuilder builder = channel.newBuilder();
            builder.startObject();
            builder.field("took", stopWatch.totalTime().getMillis());
            builder.field("timed_out", false);
            builder.startObject("hits");
            builder.field("max_score", 0);

            // In some cases there will not be enough approximate matches to return *k2* hits. For example, this could
            // be the case if the number of bits per table in the LSH model is too high, over-partioning the space.
            builder.field("total", min(k2, modifiedSortedHits.size()));
            builder.field("hits", modifiedSortedHits.subList(0, min(k2, modifiedSortedHits.size())));
            builder.endObject();
            builder.endObject();
            channel.sendResponse(new BytesRestResponse(RestStatus.OK, builder));
        };
    }

    private RestChannelConsumer handleCreateRequest(RestRequest restRequest, NodeClient client) throws IOException {

        StopWatch stopWatch = new StopWatch("StopWatch to time create request");
        logger.info("Parse request");
        stopWatch.start("Parse request");

        XContentParser xContentParser = XContentHelper.createParser(
                restRequest.getXContentRegistry(),
                DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                restRequest.content(),
                restRequest.getXContentType());
        Map<String, Object> contentMap = xContentParser.mapOrdered();
        @SuppressWarnings("unchecked")
        Map<String, Object> sourceMap = (Map<String, Object>) contentMap.get("_source");

        final String _index = (String) contentMap.get("_index");
        final String _type = (String) contentMap.get("_type");
        final String _id = (String) contentMap.get("_id");
        final String description = (String) sourceMap.get("_aknn_description");
        final Integer nbTables = (Integer) sourceMap.get("_aknn_nb_tables");
        final Integer nbBitsPerTable = (Integer) sourceMap.get("_aknn_nb_bits_per_table");
        final Integer nbDimensions = (Integer) sourceMap.get("_aknn_nb_dimensions");
        @SuppressWarnings("unchecked") final List<List<Double>> vectorSample = (List<List<Double>>) contentMap.get("_aknn_vector_sample");
        stopWatch.stop();

        logger.info("Fit LSH model from sample vectors");
        stopWatch.start("Fit LSH model from sample vectors");
        LshModel lshModel = new LshModel(nbTables, nbBitsPerTable, nbDimensions, description);
        lshModel.fitFromVectorSample(vectorSample);
        stopWatch.stop();

        logger.info("Serialize LSH model");
        stopWatch.start("Serialize LSH model");
        Map<String, Object> lshSerialized = lshModel.toMap();
        stopWatch.stop();

        logger.info("Index LSH model");
        stopWatch.start("Index LSH model");
        client.prepareIndex(_index, _type, _id)
                .setSource(lshSerialized)
                .get();
        stopWatch.stop();

        logger.info("Timing summary\n {}", stopWatch.prettyPrint());

        return channel -> {
            XContentBuilder builder = channel.newBuilder();
            builder.startObject();
            builder.field("took", stopWatch.totalTime().getMillis());
            builder.endObject();
            channel.sendResponse(new BytesRestResponse(RestStatus.OK, builder));
        };
    }

    private RestChannelConsumer handleIndexRequest(RestRequest restRequest, NodeClient client) throws IOException {

        StopWatch stopWatch = new StopWatch("StopWatch to time bulk indexing request");

        logger.info("Parse request parameters");
        stopWatch.start("Parse request parameters");
        XContentParser xContentParser = XContentHelper.createParser(
                restRequest.getXContentRegistry(),
                DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                restRequest.content(),
                restRequest.getXContentType());
        Map<String, Object> contentMap = xContentParser.mapOrdered();
        final String index = (String) contentMap.get("_index");
        final String type = (String) contentMap.get("_type");
        final String aknnURI = (String) contentMap.get("_aknn_uri");
        final boolean clear_cache = restRequest.paramAsBoolean("clear_cache", false);
        @SuppressWarnings("unchecked")
        final List<Map<String, Object>> docs = (List<Map<String, Object>>) contentMap.get("_aknn_docs");
        logger.info("Received {} docs for indexing", docs.size());
        stopWatch.stop();

        // TODO: check if the index exists. If not, create a mapping which does not index continuous values.
        // This is rather low priority, as I tried it via Python and it doesn't make much difference.

        // Clear LSH model cache if requested
        if (clear_cache) {
            lshModelCache.remove(aknnURI);
        }
        // Check if the LshModel has been cached. If not, retrieve the Aknn document and use it to populate the model.
        LshModel lshModel = initLsh(aknnURI, client);

        // Prepare documents for batch indexing.
        logger.info("Hash documents for indexing");
        stopWatch.start("Hash documents for indexing");
        BulkRequestBuilder bulkIndexRequest = client.prepareBulk();
        for (Map<String, Object> doc : docs) {
            @SuppressWarnings("unchecked")
            Map<String, Object> source = (Map<String, Object>) doc.get("_source");
            @SuppressWarnings("unchecked")
            List<Double> vector = (List<Double>) source.get(VECTOR_KEY);
            source.put(HASHES_KEY, lshModel.getVectorHashes(vector));
            bulkIndexRequest.add(client
                    .prepareIndex(index, type, (String) doc.get("_id"))
                    .setSource(source));
        }
        stopWatch.stop();

        logger.info("Execute bulk indexing");
        stopWatch.start("Execute bulk indexing");
        BulkResponse bulkIndexResponse = bulkIndexRequest.get();
        stopWatch.stop();

        logger.info("Timing summary\n {}", stopWatch.prettyPrint());

        if (bulkIndexResponse.hasFailures()) {
            logger.error("Indexing failed with message: {}", bulkIndexResponse.buildFailureMessage());
            return channel -> {
                XContentBuilder builder = channel.newBuilder();
                builder.startObject();
                builder.field("took", stopWatch.totalTime().getMillis());
                builder.field("error", bulkIndexResponse.buildFailureMessage());
                builder.endObject();
                channel.sendResponse(new BytesRestResponse(RestStatus.INTERNAL_SERVER_ERROR, builder));
            };
        }

        logger.info("Indexed {} docs successfully", docs.size());
        return channel -> {
            XContentBuilder builder = channel.newBuilder();
            builder.startObject();
            builder.field("size", docs.size());
            builder.field("took", stopWatch.totalTime().getMillis());
            builder.endObject();
            channel.sendResponse(new BytesRestResponse(RestStatus.OK, builder));
        };
    }

    @SuppressWarnings("unused")
    private RestChannelConsumer handleClearRequest(RestRequest restRequest, NodeClient client) {

        //TODO: figure out how to execute clear cache on all nodes at once;

        StopWatch stopWatch = new StopWatch("StopWatch to time clear cache");
        logger.info("Clearing LSH models cache");
        stopWatch.start("Clearing cache");
        lshModelCache.clear();
        stopWatch.stop();
        logger.info("Timing summary\n {}", stopWatch.prettyPrint());


        return channel -> {
            XContentBuilder builder = channel.newBuilder();
            builder.startObject();
            builder.field("took", stopWatch.totalTime().getMillis());
            builder.field("acknowledged", true);
            builder.endObject();
            channel.sendResponse(new BytesRestResponse(RestStatus.OK, builder));
        };
    }
}
