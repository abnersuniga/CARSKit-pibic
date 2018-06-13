package carskit.alg.cars.transformation.modeling;

import carskit.alg.baseline.cf.ItemKNN;
import carskit.alg.baseline.cf.ItemKNNUnary;
import carskit.alg.baseline.cf.UserKNN;
import carskit.data.structure.SparseMatrix;
import carskit.generic.Recommender;
import happy.coding.io.LineConfiger;
import happy.coding.io.Logs;
import librec.data.DataDAO;
import librec.data.MatrixEntry;
import java.util.*;


public class LARS extends Recommender {

    private String rec;
    private Map<Integer,Double[]> itemsLocalization;
    private Recommender recUsed;
    private Double maxUserToItemDistance, minUserToItemDistance;

    public LARS(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) throws Exception {
        super(trainMatrix, testMatrix, fold);

        LARS.algoOptions   = new LineConfiger(cf.getString("recommender"));
        this.rec = LARS.algoOptions.getString("-cf");

        recUsed = getRecommender(trainMatrix, testMatrix, fold);
        recUsed.execute();

        itemsLocalization = getItemsContext(trainMatrix);
        getMinAndMaxDistances(trainMatrix);

        System.out.println("minRec: " + recUsed.minRate);
        System.out.println("maxRec: " + recUsed.maxRate);
        System.out.println("min: " + this.minUserToItemDistance);
        System.out.println("max: " + this.maxUserToItemDistance);
    }

    @Override
    public double predict(int u, int j, int c) throws Exception {

        double recScore, p, travelPenalty, timePenalty, contextHour, contextLat, contextLong,
                itemHour, itemLat, itemLong, distance, difference;
        String[] contexts = rateDao.getContextId(c).split(",");

        contextHour = Double.parseDouble(rateDao.getContextConditionId(Integer.parseInt(contexts[1])).split(":")[2]);
        contextLat  = Double.parseDouble(rateDao.getContextConditionId(Integer.parseInt(contexts[2])).split(":")[1]);
        contextLong = Double.parseDouble(rateDao.getContextConditionId(Integer.parseInt(contexts[3])).split(":")[1]);

        itemHour = itemsLocalization.get(j)[0];
        itemLat  = itemsLocalization.get(j)[1];
        itemLong = itemsLocalization.get(j)[2];

        distance = calculateDistance(itemLat, itemLong, contextLat, contextLong);
        difference = Math.abs(contextHour - itemHour);

        travelPenalty = normalizeDistance(distance, recUsed.maxRate, 0.0);
        timePenalty = normalizeHour(difference, recUsed.maxRate, 0.0);

        p = recUsed.recommend(u, j, c);


        System.out.println(p + " - " + travelPenalty + " - " + timePenalty);
        recScore = 2*p - travelPenalty - timePenalty;

        return recScore;
    }

    private double calculateDistance(double lat1, double lon1, double lat2, double lon2){
        int R = 6371; // Radius of the earth in km
        double dLat = deg2rad(lat2-lat1);
        double dLon = deg2rad(lon2-lon1);
        double a = Math.sin(dLat/2) * Math.sin(dLat/2)
                + Math.cos(deg2rad(lat1)) * Math.cos(deg2rad(lat2)) * Math.sin(dLon/2) * Math.sin(dLon/2);
        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
        return R * c;
    }

    private double deg2rad(double deg) {
        return deg * (Math.PI/180);
    }

    private Double normalizeDistance(Double distance, Double max, Double min){
        double value = (distance - minUserToItemDistance) / (maxUserToItemDistance - minUserToItemDistance);
        return value * (max - min) + min;
    }

    private Double normalizeHour(Double diference, Double max, Double min){
        double value = (diference - 0.0) / 24.0;
        return value * (max - min) + min;
    }

    private Recommender getRecommender(SparseMatrix train, SparseMatrix test, int fold){

        Recommender recsys = null;
        switch (this.rec) {
            case "itemknn":
                recsys = new ItemKNN(train, test, fold);
                break;
            case "itemknnunary":
                recsys = new ItemKNNUnary(train, test,fold);
                break;
            case "userknn":
                recsys = new UserKNN(train, test, fold);
                break;
        }
        System.out.println("Collaborative filtering algorithm " + this.rec.toUpperCase() + "\n");

        return recsys;
    }

    private Map<Integer,Double[]> getItemsContext(SparseMatrix matrix) {
        int itemId;
        double itemLat, itemLong, itemHour;
        Double[] loc;
        Map<Integer,Double[]> itemDistances = new HashMap<Integer,Double[]>();

        for(MatrixEntry me : matrix){

            itemId = rateDao.getItemIdFromUI(me.row());
            String[] contexts = rateDao.getContextId(me.column()).split(",");

            itemHour = Double.parseDouble(rateDao.getContextConditionId(Integer.parseInt(contexts[1])).split(":")[2]);
            itemLat = Double.parseDouble(rateDao.getContextConditionId(Integer.parseInt(contexts[2])).split(":")[1]);
            itemLong = Double.parseDouble(rateDao.getContextConditionId(Integer.parseInt(contexts[3])).split(":")[1]);

            loc = new Double[3];
            loc[0] = itemHour;
            loc[1] = itemLat;
            loc[2] = itemLong;

            itemDistances.put(itemId, loc);
        }

        return itemDistances;
    }

    private void getMinAndMaxDistances(SparseMatrix matrix) {
        int itemIdI,itemIdJ;
        double itemLatI, itemLongI, itemLatJ, itemLongJ,distance;
        boolean first = true;
        String[] contexts;

        for(MatrixEntry mei : matrix){

            itemIdI = rateDao.getItemIdFromUI(mei.row());
            contexts = rateDao.getContextId(mei.column()).split(",");

            itemLatI = Double.parseDouble(rateDao.getContextConditionId(Integer.parseInt(contexts[2])).split(":")[1]);
            itemLongI = Double.parseDouble(rateDao.getContextConditionId(Integer.parseInt(contexts[3])).split(":")[1]);

            for(MatrixEntry mej : matrix){

                itemIdJ = rateDao.getItemIdFromUI(mej.row());
                contexts = rateDao.getContextId(mej.column()).split(",");

                itemLatJ = Double.parseDouble(rateDao.getContextConditionId(Integer.parseInt(contexts[2])).split(":")[1]);
                itemLongJ = Double.parseDouble(rateDao.getContextConditionId(Integer.parseInt(contexts[3])).split(":")[1]);


                distance = calculateDistance(itemLatI, itemLongI, itemLatJ, itemLongJ);

                if(first || distance > maxUserToItemDistance){
                    maxUserToItemDistance = distance;
                }

                if(first || distance < minUserToItemDistance){
                    minUserToItemDistance = distance;
                }

                first = false;
            }
        }
    }

}


