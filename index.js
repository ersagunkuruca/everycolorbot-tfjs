var Twitter = require("twitter");
var fs = require("fs");

var config = require("./config");

var client = new Twitter(config);
var count = 200;
var params = {
  screen_name: "everycolorbot",
  //since_id: 1,
  //max_id: "1095510449238536192",
  count: count,
  trim_user: true,
  exclude_replies: true,
  include_rts: false
};
var tweet_ids = {};
var all_tweets = [];
function get_tweets() {
  console.log("getting tweets ", params.max_id);
  client.get("statuses/user_timeline", params, function(
    error,
    tweets,
    response
  ) {
    if (!error) {
      console.log(Object.keys(tweets[0]));
      if (tweets.length === count) {
        tweets.forEach(a => {
          if (!(a.id_str in tweet_ids)) {
            var newA = {
              id: a.id_str,
              //text: a.text,
              color: a.text.substring(0, 8),
              favs: a.favorite_count,
              rts: a.retweet_count
            };
            tweet_ids[a.id_str] = true;
            all_tweets.push(newA);
          }
        });
        params.max_id = tweets[tweets.length - 1].id_str;
        get_tweets();
      } else {
        fs.writeFile(
          "everycolorbot.json",
          JSON.stringify(all_tweets, null, " "),
          "utf8",
          function() {
            console.log("done");
          }
        );
      }
    } else {
      console.log(error);
    }
  });
}
get_tweets();
