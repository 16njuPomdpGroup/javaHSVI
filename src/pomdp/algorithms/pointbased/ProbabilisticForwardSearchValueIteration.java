package pomdp.algorithms.pointbased;

import pomdp.algorithms.ValueIteration;
import pomdp.environments.FactoredPOMDP;
import pomdp.environments.POMDP;
import pomdp.utilities.*;
import pomdp.utilities.factored.FactoredBeliefState;
import pomdp.valuefunction.LinearValueFunctionApproximation;

import java.util.*;

public class ProbabilisticForwardSearchValueIteration extends ValueIteration {
    protected int m_iLimitedBeliefMDPState;
    protected int m_iLimitedBeliefObservation;
    protected LinearValueFunctionApproximation m_vDetermisticPOMDPValueFunction;
    protected BeliefState m_bsDeterministicPOMDPBeliefState;
    protected HeuristicPolicy m_hpPolicy;
    protected int m_iDepth;
    protected int m_iIteration, m_iInnerIteration;
    protected long m_lLatestADRCheck, m_cTimeInADR, m_lCPUTimeTotal, m_lIterationStartTime;
    protected Pair m_pComputedADRs;
    protected int[] m_aiStartStates;
    protected SortedMap<Double,Integer>[][] m_amNextStates;

    //点集B
    protected BeliefStateVector<BeliefState> vBeliefPoints = new BeliefStateVector<BeliefState>();
    protected static double OBSERVATION_THERSHOLD = 0.01;
    long startTime = 0;
    long endTime = 0;

    public ProbabilisticForwardSearchValueIteration(POMDP pomdp){
        super( pomdp );
        m_iDepth = 0;
        m_iIteration = 0;
        m_iInnerIteration = 0;
        m_lLatestADRCheck = 0;
        m_cTimeInADR = 0;
        m_lCPUTimeTotal = 0;
        m_lIterationStartTime = 0;
        m_pComputedADRs = null;
        m_aiStartStates = null;
        m_vfMDP = null;
        m_bsDeterministicPOMDPBeliefState = null;
        m_vDetermisticPOMDPValueFunction = null;
        m_iLimitedBeliefObservation = -1;
        initHeuristic();
        //m_vLabeledBeliefs = new Vector<BeliefState>();
    }

    private void initHeuristic(){
        long lBefore = JProf.getCurrentThreadCpuTimeSafe(), lAfter = 0;
        m_vfMDP = m_pPOMDP.getMDPValueFunction();
        m_vfMDP.valueIteration( 1000, ExecutionProperties.getEpsilon() );
        lAfter = JProf.getCurrentThreadCpuTimeSafe();
        Logger.getInstance().log( "PFSVI", 0, "initHeurisitc", "Initialization time was " + ( lAfter - lBefore ) / 1000000 );
    }

    public void valueIteration( int cMaxSteps, double dEpsilon, double dTargetValue, int maxRunningTime, int numEvaluations) {

        //public void valueIteration( int cMaxSteps, double dEpsilon, double dTargetValue ){
        int iIteration = 0;
        boolean bDone = false;
        Pair pComputedADRs = new Pair();
        double dMaxDelta = 0.0;
        String sMsg = "";

        long lStartTime = System.currentTimeMillis(), lCurrentTime = 0;
        long lCPUTimeBefore = 0, lCPUTimeAfter = 0;
        Runtime rtRuntime = Runtime.getRuntime();

        long cDotProducts = AlphaVector.dotProductCount(), cVnChanges = 0, cStepsWithoutChanges = 0;
        m_cElapsedExecutionTime = 0;
        m_lCPUTimeTotal = 0;

        sMsg = "Starting " + getName() + " target ADR = " + round( dTargetValue, 3 );
        Logger.getInstance().log( "PFSVI", 0, "VI", sMsg );

        //initStartStateArray();
        m_pComputedADRs = new Pair();

        for( iIteration = 0 ; ( iIteration < cMaxSteps ) && !bDone ; iIteration++ ){
            lStartTime = System.currentTimeMillis();
            lCPUTimeBefore = JProf.getCurrentThreadCpuTimeSafe();
            AlphaVector.initCurrentDotProductCount();
            cVnChanges = m_vValueFunction.getChangesCount();
            m_iIteration = iIteration;
            m_iInnerIteration = 0;
            m_lLatestADRCheck = lCPUTimeBefore;
            m_cTimeInADR = 0;
            m_lIterationStartTime = lCPUTimeBefore;
            dMaxDelta = improveValueFunction();
            lCPUTimeAfter = JProf.getCurrentThreadCpuTimeSafe();
            lCurrentTime = System.currentTimeMillis();
            m_cElapsedExecutionTime += ( lCurrentTime - lStartTime - m_cTimeInADR );
            m_cCPUExecutionTime += ( lCPUTimeAfter - lCPUTimeBefore - m_cTimeInADR ) / 1000000;
            m_lCPUTimeTotal += lCPUTimeAfter - lCPUTimeBefore - m_cTimeInADR;

            if( m_bTerminate )
                bDone = true;


            if( ExecutionProperties.getReportOperationTime() ){
                try{
                    sMsg = "G: - operations " + AlphaVector.getGComputationsCount() + " avg time " +
                            AlphaVector.getAvgGTime();
                    Logger.getInstance().log( "PFSVI", 0, "VI", sMsg );

                    if( m_pPOMDP.isFactored() && ((FactoredPOMDP) m_pPOMDP).getBeliefType() == FactoredPOMDP.BeliefType.Factored ){
                        sMsg = "Tau: - operations " + FactoredBeliefState.getTauComputationCount() + " avg time " +
                                FactoredBeliefState.getAvgTauTime();
                        Logger.getInstance().log( "PFSVI", 0, "VI", sMsg );

                    }
                    else{
                        sMsg = "Tau: - operations " + m_pPOMDP.getBeliefStateFactory().getTauComputationCount() + " avg time " +
                                m_pPOMDP.getBeliefStateFactory().getAvgTauTime();
                        Logger.getInstance().log( "PFSVI", 0, "VI", sMsg );
                    }
                    sMsg = "dot product - avg time = " + AlphaVector.getCurrentDotProductAvgTime();
                    Logger.getInstance().log( "PFSVI", 0, "VI", sMsg );
                    sMsg = "avg belief state size " + m_pPOMDP.getBeliefStateFactory().getAvgBeliefStateSize();
                    Logger.getInstance().log( "PFSVI", 0, "VI", sMsg );
                    sMsg = "avg alpha vector size " + m_vValueFunction.getAvgAlphaVectorSize();
                    Logger.getInstance().log( "PFSVI", 0, "VI", sMsg );
                    AlphaVector.initCurrentDotProductCount();
                }
                catch( Exception e ){
                    Logger.getInstance().logln( e );
                }
            }
            if( ( ( m_lCPUTimeTotal  / 1000000000 ) >= 5 ) && ( iIteration >= 10 ) && ( iIteration % 5 == 0 ) &&
                    m_vValueFunction.getChangesCount() > cVnChanges &&
                    m_vValueFunction.size() > 5 ){



                cStepsWithoutChanges = 0;
                bDone |= checkADRConvergence( m_pPOMDP, dTargetValue, pComputedADRs );

                sMsg = "PFSVI: Iteration " + iIteration +
                        "; |Vn| = " + m_vValueFunction.size() +
                        "; simulated_ADR " + round( ((Number) pComputedADRs.first()).doubleValue(), 3 ) +
                        "; filtered_ADR " + round( ((Number) pComputedADRs.second()).doubleValue(), 3 ) +
                        "; max_delta " + round( dMaxDelta, 3 ) +
                        "; depth " + m_iDepth +
                        "; V(b0) " + round( m_vValueFunction.valueAt( m_pPOMDP.getBeliefStateFactory().getInitialBeliefState() ), 2 ) +
                        "; time " + 	( lCurrentTime - lStartTime ) +
                        "; total_ime " + 	m_cElapsedExecutionTime +
                        "; CPU_time " + ( lCPUTimeAfter - lCPUTimeBefore - m_cTimeInADR ) / 1000000000 +
                        "; CPU_total " + m_lCPUTimeTotal  / 1000000000 +
                        "; #backups " + m_cBackups +
                        "; V_changes " + m_vValueFunction.getChangesCount() +
                        "; #dot product " + AlphaVector.dotProductCount() +
                        "; |BS| " + m_pPOMDP.getBeliefStateFactory().getBeliefStateCount() +
                        "; memory: (" +
                        "; total " + rtRuntime.totalMemory() / 1000000 +
                        "; free " + rtRuntime.freeMemory() / 1000000 +
                        "; max " + rtRuntime.maxMemory() / 1000000 +
                        ")";
                Logger.getInstance().log( "PFSVI", 0, "VI", sMsg );
            }
            else{
                if( cVnChanges == m_vValueFunction.getChangesCount() ){
                    cStepsWithoutChanges++;
                    //if( cStepsWithoutChanges == 10 ){
                    //	bDone = true;
                    //}
                }
                sMsg = "PFSVI: Iteration " + iIteration +
                        "; |Vn| = " + m_vValueFunction.size() +
                        "; time " + 	( lCurrentTime - lStartTime ) +
                        "; total_ime " + 	m_cElapsedExecutionTime +
                        "; V_changes " + m_vValueFunction.getChangesCount() +
                        "; max_delta " + round( dMaxDelta, 3 ) +
                        "; depth " + m_iDepth +
                        "; V(b0) " + round( m_vValueFunction.valueAt( m_pPOMDP.getBeliefStateFactory().getInitialBeliefState() ), 2 ) +
                        "; CPU_time " + ( lCPUTimeAfter - lCPUTimeBefore ) / 1000000000 +
                        "; CPU_total " + m_lCPUTimeTotal  / 1000000000 +
                        "; #backups " + m_cBackups +
                        "; |BS| " + m_pPOMDP.getBeliefStateFactory().getBeliefStateCount() +
                        "; memory: (" +
                        " total " + rtRuntime.totalMemory() / 1000000 +
                        " free " + rtRuntime.freeMemory() / 1000000 +
                        " max " + rtRuntime.maxMemory() / 1000000 +
                        ")";
                Logger.getInstance().log( "PFSVI", 0, "VI", sMsg );


            }

        }
        m_bConverged = true;

        m_cDotProducts = AlphaVector.dotProductCount() - cDotProducts;
//        m_cElapsedExecutionTime /= 1000;
        m_cCPUExecutionTime /= 1000;

        sMsg = "Finished " + getName() + " - time : " + m_cElapsedExecutionTime + /*" |BS| = " + vBeliefPoints.size() +*/
                " |V| = " + m_vValueFunction.size() +
                " backups = " + m_cBackups +
                " GComputations = " + AlphaVector.getGComputationsCount() +
                " Dot products = " + m_cDotProducts;
        Logger.getInstance().log( "PFSVI", 0, "VI", sMsg );
    }
    protected double forwardSearch( int iState, BeliefState bsCurrent, int iDepth ){
        String sMsg;
        double dDelta = 0.0, dNextDelta = 0.0;
        int iNextState = 0, iHeuristicAction = 0, iPOMDPAction = 0, iObservation = 0;
        BeliefState bsNext = null;
        BeliefState bsTemp = null;
        AlphaVector avBackup = null, avMax = null;
        double dPreviousValue = 0.0, dNewValue = 0.0;

        double[] actionPr = new double[m_cActions];
        double[] nextStatePr = new double[m_cStates];
        for(int action = 0; action < m_cActions; action++) {
            actionPr[action] = 0.0;
        }


        if( m_bTerminate )
            return 0.0;

        if( ( m_pPOMDP.terminalStatesDefined() && isTerminalState( iState ) )
                ||	( iDepth >= 100 ) ){
            m_iDepth = iDepth;
            Logger.getInstance().logln( "Ended at depth " + iDepth + ". isTerminalState(" + iState + ")=" + isTerminalState( iState ) );
        }
        else{
            //创建actionPr
            for(int state = 0; state < m_cStates; state++) {
                actionPr[m_vfMDP.getAction(state)] += bsCurrent.valueAt(state);
            }

            iHeuristicAction = aliasMethod(actionPr);


            //创建nextStatePr
            for(int nextState = 0; nextState < m_cStates; nextState++) {
                nextStatePr[nextState] = 0.0;
                for(int preState = 0; preState < m_cStates; preState++) {
                    nextStatePr[nextState] += (bsCurrent.valueAt(preState) * m_pPOMDP.tr(preState, iHeuristicAction, nextState));
                }
            }

            iNextState = aliasMethod(nextStatePr);


            //
            List<Integer> observationList = new ArrayList<Integer>();
            for(int observation = 0; observation < m_cObservations; observation++) {
                if (m_pPOMDP.O(iHeuristicAction, iNextState, observation) > OBSERVATION_THERSHOLD) {
                    observationList.add(observation);
                }
            }
            Random random = new Random();
            iObservation = observationList.get(random.nextInt(observationList.size()));

            bsNext = bsCurrent.nextBeliefState( iHeuristicAction, iObservation );


            if( bsNext == null || bsNext.equals( bsCurrent ) ){ // || isTerminalState(iNextState) ){
                m_iDepth = iDepth;
            }
            else{
                //vBeliefPoints.add(bsNext);
                dNextDelta = forwardSearch( iNextState, bsNext, iDepth + 1 );
            }
        }


        if( true ){
            BeliefState bsDeterministic = getDeterministicBeliefState( iState );
            avBackup = backup( bsDeterministic, iHeuristicAction );
            dPreviousValue = m_vValueFunction.valueAt( bsDeterministic );
            dNewValue = avBackup.dotProduct( bsDeterministic );
            dDelta = dNewValue - dPreviousValue;

            if( dDelta > ExecutionProperties.getEpsilon() ){
                m_vValueFunction.addPrunePointwiseDominated( avBackup );
            }
        }
        avBackup = backup( bsCurrent );

        dPreviousValue = m_vValueFunction.valueAt( bsCurrent );
        dNewValue = avBackup.dotProduct( bsCurrent );
        dDelta = dNewValue - dPreviousValue;
        avMax = m_vValueFunction.getMaxAlpha( bsCurrent );

        if( dDelta > 0.0 ){
            m_vValueFunction.addPrunePointwiseDominated( avBackup );
        }
        else{
            avBackup.release();
        }

        return Math.max( dDelta, dNextDelta );
    }

    protected boolean isTerminalState( int iState ){
        return m_pPOMDP.isTerminalState( iState );
    }
    protected BeliefState getDeterministicBeliefState( int iState ){
        return m_pPOMDP.getBeliefStateFactory().getDeterministicBeliefState( iState );
    }
    private int getAction( int iState, BeliefState bs ){
        if( m_rndGenerator.nextDouble() < 0.9 )
            return m_vfMDP.getAction( iState );
        return m_rndGenerator.nextInt( m_cActions );
    }
    private int getObservation( int iStartState, int iAction, int iEndState ){
        return m_pPOMDP.observe( iAction, iEndState );
    }
    private int selectNextState( int iState, int iAction ) {
        return m_pPOMDP.execute( iAction, iState );
    }

    private void removeNextState( int iState, int iAction, int iNextState ){
        m_amNextStates[iState][iAction].remove( m_amNextStates[iState][iAction].lastKey() );
    }

    protected int getNextState( int iAction, int iState ){
        int iNextState = -1;
        double dTr = 0.0, dValue = 0.0;

        if( m_amNextStates[iState][iAction] == null )
            m_amNextStates[iState][iAction] = new TreeMap<Double,Integer>();
        if( m_amNextStates[iState][iAction].isEmpty() ){
            Iterator itNonZero = m_pPOMDP.getNonZeroTransitions( iState, iAction );
            Map.Entry e = null;
            String sDescription = "";
            while( itNonZero.hasNext() ){
                e = (Map.Entry) itNonZero.next();
                iNextState = ((Number)e.getKey()).intValue();
                dTr = ((Number)e.getValue()).doubleValue();
                dValue = m_vfMDP.getValue( iNextState );
                sDescription += "V(" + iNextState + ") = " + round( dValue, 3 ) + ", ";
                m_amNextStates[iState][iAction].put( dValue, iNextState );
            }
        }
        iNextState = m_amNextStates[iState][iAction].get( m_amNextStates[iState][iAction].lastKey() );
        return iNextState;
    }

    protected void initStartStateArray(){
        int cStates = m_pPOMDP.getStartStateCount(), iState = 0;
        Iterator<Map.Entry<Integer,Double>> itStartStates = m_pPOMDP.getStartStates();
        Map.Entry<Integer,Double> e = null;
        m_aiStartStates = new int[cStates];
        for( iState = 0 ; iState < cStates ; iState++ ){
            e = itStartStates.next();
            m_aiStartStates[iState] = e.getKey();
        }
        if( m_amNextStates == null ){
            m_amNextStates = new SortedMap[m_cStates][m_cActions];
        }
    }

    protected int chooseStartState(){
        int cStates = m_pPOMDP.getStartStateCount(), iState = 0, iMaxValueState = -1;
        double dValue = 0.0, dMaxValue = MIN_INF;
        for( iState = 0 ; iState < cStates ; iState++ ){
            if( m_aiStartStates[iState] != -1 ){
                dValue = m_vfMDP.getValue( iState );
                if( dValue > dMaxValue ){
                    dMaxValue = dValue;
                    iMaxValueState = iState;
                }
            }
        }
        if( iMaxValueState == -1 ){
            initStartStateArray();
            return chooseStartState();
        }
        iState = m_aiStartStates[iMaxValueState];
        m_aiStartStates[iMaxValueState] = -1;
        return iState;
    }


    protected double improveValueFunction(){
        int iInitialState = -1;
        do{
            iInitialState = m_pPOMDP.chooseStartState();
        }while( m_pPOMDP.isTerminalState( iInitialState ) );
        BeliefState bsInitial = m_pPOMDP.getBeliefStateFactory().getInitialBeliefState();
        Logger.getInstance().logln( "Starting at state " + m_pPOMDP.getStateName( iInitialState ) );
        m_iDepth = 0;
        Logger.getInstance().logln( "Begin improve" );
        double dDelta = forwardSearch( iInitialState, bsInitial, 0 );
        Logger.getInstance().logln( "End improve, |V| = " +
                m_vValueFunction.size() + ", delta = " + dDelta );
        return dDelta;
    }


    protected int aliasMethod(double[] probabilities) {
//        Random random = new Random();
//        return random.nextInt(probabilities.length);
        /* Allocate space for the probability and alias tables. */
        double[] probability = new double[probabilities.length];
        int[] alias = new int[probabilities.length];

        /* Compute the average probability and cache it for later use. */
        final double average = 1.0 / probabilities.length;

        /* Make a copy of the probabilities list, since we will be making
         * changes to it.
         */
//        probabilities = new ArrayList<Double>(probabilities);

        /* Create two stacks to act as worklists as we populate the tables. */
        Deque<Integer> small = new ArrayDeque<Integer>();
        Deque<Integer> large = new ArrayDeque<Integer>();

        /* Populate the stacks with the input probabilities. */
        for (int i = 0; i < probabilities.length; ++i) {
            /* If the probability is below the average probability, then we add
             * it to the small list; otherwise we add it to the large list.
             */
            if (probabilities[i] >= average)
                large.add(i);
            else
                small.add(i);
        }

        /* As a note: in the mathematical specification of the algorithm, we
         * will always exhaust the small list before the big list.  However,
         * due to floating point inaccuracies, this is not necessarily true.
         * Consequently, this inner loop (which tries to pair small and large
         * elements) will have to check that both lists aren't empty.
         */
        while (!small.isEmpty() && !large.isEmpty()) {
            /* Get the index of the small and the large probabilities. */
            int less = small.removeLast();
            int more = large.removeLast();

            /* These probabilities have not yet been scaled up to be such that
             * 1/n is given weight 1.0.  We do this here instead.
             */
            probability[less] = probabilities[less] * probabilities.length;
            alias[less] = more;

            /* Decrease the probability of the larger one by the appropriate
             * amount.
             */
            probabilities[more] = (probabilities[more] + probabilities[less]) - average;

            /* If the new probability is less than the average, add it into the
             * small list; otherwise add it to the large list.
             */
            if (probabilities[more] >= 1.0 / probabilities.length)
                large.add(more);
            else
                small.add(more);
        }

        /* At this point, everything is in one list, which means that the
         * remaining probabilities should all be 1/n.  Based on this, set them
         * appropriately.  Due to numerical issues, we can't be sure which
         * stack will hold the entries, so we empty both.
         */
        while (!small.isEmpty())
            probability[small.removeLast()] = 1.0;
        while (!large.isEmpty())
            probability[large.removeLast()] = 1.0;


        /* Generate a fair die roll to determine which column to inspect. */
        Random random = new Random();
        int column = random.nextInt(probability.length);

        /* Generate a biased coin toss to determine which option to pick. */
        boolean coinToss = random.nextDouble() < probability[column];

        /* Based on the outcome, return either the column or its alias. */
        return coinToss? column : alias[column];
    }
}
