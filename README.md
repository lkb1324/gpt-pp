# gpt-pp
this is for ChatGPT popularity prediction work's cooperation &amp; communication

Each .csv profile is blogs of one hashtag from 2023.3-2023.7, which is actually all the blogs of one hashtag, considering the short lifespan of hashtags on Weibo.

In csv files, one piece of data refers to one blog talking about the hashtag which is displayed in the file name. The profiles of a blog include:

{
mid: unique id of each blog,
**publish_time: publish_time,**
user_name: name of user who has published this blog,
**user_link: link of user's homepage,**
content: content of the blog,
blog_length: the number of str in this blog,
@_number: the number of @ in this blog,
forward_num: times the blog has been forwarded,
comment_num: like above,
like_num: like above,
total_forward: sum of the three above,
**user_id: unique id of each user, also included in user_link,**
Weibo_numer: number of blogs the user has posted,
verification type: verify type of user on Weibo, may include not verified, celebrity, official, and so on,
mbrank
mbtype
creat_at: there is something wrong with they three, don't bother it for now,
sunshine: sunshine credit level of user.
}

You may use user_link or user_id and publish_time to create directed or undirected graph of users talking about same hashtag and the time sequence they talk about it according to your need.





