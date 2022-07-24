CREATE DEFINER=`root`@`%` TRIGGER `uodaterating` AFTER INSERT ON `Review` FOR EACH ROW Begin
	set @numreview = (select count(reviewID) From Review Where userID =new.userID Group by userID);
	set @nummoviereT = (select count(reviewID) From Review Where movieID =new.movieID and source = 'Rotten Tomatoes' Group by movieID);
	set @nummoviereI = (select count(reviewID) From Review Where movieID =new.movieID and source = 'IMDB' Group by movieID);
	set @sumratingI = (select sum(rating) From Review Where movieID =new.movieID and source = 'IMDB' group by movieID);
	set @sumratingT = (select sum(rating) From Review Where movieID =new.movieID and source = 'Rotten Tomatoes' group by movieID);
	IF (@numreview) >=5 AND NEW.source = 'IMDB' THEN
		UPDATE `schema`.Movie
		set ratingFromIMDB = @sumratingI/@nummoviereI
		where movieID = new.movieID;
	elseif (@numreview) >=5 AND NEW.source = 'Rotten Tomatoes' THEN
		UPDATE `schema`.Movie
		set ratingFromTomato = @sumratingT/@nummoviereT
		where movieID = new.movieID;
	end if;
end;