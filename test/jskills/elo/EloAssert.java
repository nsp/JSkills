﻿using Moserware.Skills;
using Moserware.Skills.Elo;
using NUnit.Framework;

namespace UnitTests.Elo
{
    internal static class EloAssert
    {
        private final double ErrorTolerance = 0.1;

        public static void AssertChessRating(TwoPlayerEloCalculator calculator,
                                             double player1BeforeRating,
                                             double player2BeforeRating,
                                             PairwiseComparison player1Result,
                                             double player1AfterRating,
                                             double player2AfterRating)
        {
            player1 = new Player(1);
            player2 = new Player(2);

            teams = Teams.Concat(
                new Team(player1, new EloRating(player1BeforeRating)),
                new Team(player2, new EloRating(player2BeforeRating)));

            chessGameInfo = new GameInfo(1200, 0, 200, 0, 0);

            result = calculator.CalculateNewRatings(chessGameInfo, teams,
                (player1Result == PairwiseComparison.Win) ? new[] { 1, 2 } :
                (player1Result == PairwiseComparison.Lose) ? new[] { 2, 1 } :
                new[] { 1, 1 });


            assertEquals(player1AfterRating, result[player1].Mean, ErrorTolerance);
            assertEquals(player2AfterRating, result[player2].Mean, ErrorTolerance);
        }
    }
}
