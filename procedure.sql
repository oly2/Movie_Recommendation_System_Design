CREATE DEFINER=`root`@`%` PROCEDURE `Result15`()
BEGIN
	DECLARE
		varMovieID INT;
	DECLARE
		varNumreview INT;
	DECLARE
		varTomatoRating INT;
	DECLARE
		varIMDBRating INT;
	DECLARE
		varMostmovieActorid INT;
	DECLARE
		varYeargroup INT;
	DECLARE
		varreleaseDate INT;
	DECLARE
		loop_exit Boolean DEFAULT FALSE;
	DECLARE
		cur CURSOR FOR ( SELECT movieID, releaseDate FROM Movie where ratingFromIMDB > 7 and ratingFromTomato >90);
	DECLARE
		CONTINUE HANDLER FOR NOT FOUND 
		SET loop_exit = TRUE;
	DROP TABLE
	IF
		EXISTS Yearsmovie;
	CREATE TABLE Yearsmovie ( MovieID INT PRIMARY KEY, Numreview INT, IMDBrating INT, Tomatorating INT, Mostmovieactor INT, Yeargroup INT );
	OPEN cur;
	cloop :
	LOOP
			FETCH cur INTO varMovieID,
			varreleaseDate;
		IF
			loop_exit THEN
				LEAVE cloop;
			
		END IF;
		IF
			( varreleaseDate >= 2010 ) THEN
				
				SET varYeargroup = 2010;
			
			ELSEIF ( varreleaseDate < 2010 AND varreleaseDate >= 2000 ) THEN
			
			SET varYeargroup = 2000;
			
			ELSEIF ( varreleaseDate < 2000 AND varreleaseDate >= 1990 ) THEN
			
			SET varYeargroup = 1990;
				
		END IF;
		
		SET varMostmovieActorid = (
			SELECT
				actorID 
				FROM
					(
					SELECT
						actorID,
						count( movieID ) AS nummovie 
					FROM
						( SELECT actorID FROM Actor NATURAL JOIN Act WHERE movieID = varMovieID ) AS findactor
						NATURAL JOIN Act
						NATURAL JOIN Movie 
					GROUP BY
						actorID 
					) AS countmovie 
				 order by nummovie
				 limit 1);
			
			SET varNumreview = ( SELECT count( reviewID ) FROM Review NATURAL JOIN Movie WHERE movieID = varMovieID GROUP BY movieID );
			
			SET varIMDBRating = ( SELECT ratingFromIMDB FROM Movie WHERE movieID = varMovieID );
			
			SET varTomatoRating = ( SELECT ratingFromTomato FROM Movie WHERE movieID = varMovieID );
			INSERT INTO Yearsmovie
			VALUES
				( varMovieID, varNumreview, varIMDBRating, varTomatoRating, varMostmovieActorid, varYeargroup );
			
		END LOOP cloop;
		CLOSE cur;
		SELECT
			MovieID,
			IMDBrating,
			Tomatorating,
			Mostmovieactor,
			Yeargroup 
		FROM
			Yearsmovie Ym
		where Numreview>100	
		ORDER BY
			Ym.Tomatorating;
	
END