"""Second half of the extra Board of Governors rules (see league_rule_catalog_ext)."""

DATA_B = """
RVX001|revenue|National Streaming Rights Sale|The league sells national streaming rights: +2% revenue.|rev_all=2.0,fan=0.2||0.15
RVX002|revenue|In-Market Streaming Allowed|Regional TV blackouts lifted.|rev_all=0.8,fan=0.6,rev_large=-0.3|small=0.05|0.08
RVX003|revenue|Arena Sportsbook Licences|Sportsbooks open inside arenas.|rev_all=1.5,fan=-0.5|poor=0.15|0.0
RVX004|revenue|Third-Period Alcohol Sales|Beer sales run through the third period.|rev_all=0.4,fan=0.2||0.08
RVX005|revenue|Family Section Mandate|Every arena holds a low-price family section.|rev_all=-0.4,fan=1.0||0.0
RVX006|revenue|Premium Seating Expansion|More luxury seating in every arena.|rev_all=1.0,fan=-0.3|large=0.1|0.05
RVX007|revenue|Ticket Resale Cap|Resale prices capped at face value plus 10%.|fan=0.8,rev_all=-0.2||0.02
RVX008|revenue|Official Resale Marketplace|The league runs the resale market.|rev_all=0.6,fan=-0.1||0.1
RVX009|revenue|Revenue Floor Guarantee|Every club guaranteed 80% of median revenue.|revenue_share=1.2|small=0.3,poor=0.2,large=-0.3|-0.05
RVX010|revenue|Top-Market Contribution Cap|Large markets' sharing payments capped.|revenue_share=-1.0,rev_large=0.4|large=0.3,small=-0.3|-0.05
RVX011|revenue|Pooled Merchandise|Jersey sales split evenly across clubs.|rev_small=1.0,rev_large=-0.8,star_rev=-2.0|small=0.25,large=-0.3|-0.05
RVX012|revenue|Local Sponsorship Freedom|Clubs may sell any category of local sponsor.|rev_large=0.8,rev_all=0.4|large=0.15|0.05
RVX013|revenue|Naming Rights Pool|Part of naming-rights money is shared.|revenue_share=0.4,rev_large=-0.3|small=0.15,large=-0.15|0.0
RVX014|revenue|Digital Fan Tokens|Digital fan tokens sold league-wide.|rev_all=0.6,fan=-0.4||-0.02
RVX015|revenue|Season Ticket Price Cap|Season tickets can rise at most 3% a year.|rev_all=-0.6,fan=0.8|rich=0.05,poor=-0.15|-0.02
RVX016|revenue|Premium Camera Package|Sky cams and rail cams in every arena.|rev_all=0.5,fan=0.4,opex=0.2||0.08
RVX017|revenue|Women's Pro Joint Marketing|Joint marketing with the women's pro league.|rev_all=0.3,fan=0.5||0.1
RVX018|revenue|Youth Ticket Subsidy|The league subsidises under-18 tickets.|fan=0.7,opex=0.2||0.06
RVX019|revenue|All-In Ticket Pricing|All-in ticket prices required.|fan=0.5,rev_all=-0.2||0.06
RVX020|revenue|Sharing Effort Clause|Sharing money depends on ticket sales effort.|revenue_share=-0.3,fan=0.2|large=0.1,poor=-0.1|0.0
RVX021|revenue|Visitors Share Playoff Gates|Visiting clubs get 15% of playoff gates.|playoff_rev=-10.0,revenue_share=0.2|contender=-0.15,rebuilder=0.05|0.0
RVX022|revenue|Playoff Ticket Surcharge|Playoff tickets carry a league surcharge.|playoff_rev=12.0,fan=-0.3|contender=0.1|0.0
RVX023|revenue|Star Jersey Royalty|Stars' clubs keep more of their jersey sales.|star_rev=10.0,rev_small=-0.2|large=0.05|0.0
RVX024|revenue|Likeness Licensing Pool|Player likeness money shared with clubs.|star_rev=5.0,rev_all=0.3||0.05
RVX025|revenue|League Concession Partner|A league-wide concession partnership.|rev_all=0.5,opex=-0.2||0.1
RVX026|revenue|Cashless Arenas|All arenas go cashless.|rev_all=0.3,fan=-0.3,opex=-0.2||0.02
RVX027|revenue|Season Ticket Playoff Perk|Season-ticket holders get free playoff seats.|playoff_rev=-6.0,fan=0.8||0.0
RVX028|revenue|Dynamic Parking Prices|Arena parking priced by demand.|rev_all=0.3,fan=-0.4||0.0
RVX029|revenue|Small-Market TV Subsidy|National TV subsidises small-market broadcasts.|rev_small=1.5,rev_large=-0.5|small=0.25,large=-0.2|0.0
RVX030|revenue|Broadcast Rights Pooling|All local TV pooled and split evenly.|revenue_share=2.0,rev_large=-1.0,rev_small=1.0|small=0.35,large=-0.4|-0.08
RVX031|revenue|Suite Tax Lobbying|The league lobbies for luxury-suite tax breaks.|rev_large=0.5,opex=0.1|large=0.1|0.05
RVX032|revenue|Free Over-the-Air Games|Ten national games a season on free TV.|fan=0.8,rev_all=-0.3,star_rev=2.0||0.05
RVX033|revenue|Pay-Per-View Rivalries|Select rivalry games sold as pay-per-view.|rev_all=0.6,fan=-0.6|poor=0.1|-0.03
RVX034|revenue|International TV Push|Overseas rights sold aggressively: +1% revenue.|rev_all=1.0,star_rev=2.0||0.1
RVX035|revenue|Kids Free Sundays|Children free at Sunday games.|fan=0.9,rev_all=-0.4||0.0
RVX036|revenue|Year-Round Team Stores|Year-round team stores in every arena.|rev_all=0.5,opex=0.2||0.06
RVX037|revenue|Sharing Sunset Clause|The sharing pool shrinks every year.|revenue_share=-0.8|large=0.25,small=-0.25|-0.03
RVX038|revenue|Loss Insurance|The league insures clubs against losses above $15M.|opex=0.3,relocation_ease=-0.03|poor=0.2,rich=-0.1|0.0
RVX039|revenue|Sharing Audit|Independent audits of shared revenue.|revenue_share=0.1,fan=0.1||0.1
RVX040|revenue|Practice Ticket Sales|Clubs sell tickets to practices.|rev_all=0.2,fan=0.2||0.1
RVX041|revenue|Team Content Studios|Every club runs a content studio.|rev_all=0.4,opex=0.3,fan=0.3||0.02
RVX042|revenue|Tracking Data Sales|The league sells live tracking data to bettors.|rev_all=0.8,fan=-0.3||0.05
RVX043|revenue|Concession Revenue Sharing|10% of concessions go into sharing.|revenue_share=0.5,rev_large=-0.2|small=0.15,large=-0.15|0.0
RVX044|revenue|Playoff Revenue Split Evenly|All playoff revenue split among every club.|playoff_rev=-30.0,revenue_share=0.8|rebuilder=0.2,contender=-0.3|-0.05
RVX045|revenue|Franchise Tax Credit Lobbying|The league lobbies for franchise tax credits.|value_growth=0.5|rich=0.1|0.05
TDX001|trades|Deadline Moved to March 20|The trade deadline moves two weeks later.|trade_volume=6.0,fan=0.3|contender=0.1|0.02
TDX002|trades|January Trade Deadline|The deadline moves to late January.|trade_volume=-10.0,fan=-0.3|rebuilder=-0.1|-0.05
TDX003|trades|No Trade Deadline|Trades allowed until the final day of the season.|trade_volume=15.0,trade_demand_rate=5.0,fan=0.2|all=-0.06|-0.08
TDX004|trades|Five Retained-Salary Slots|Clubs may carry five retained contracts.|retention_slots=2|contender=0.15,rebuilder=0.15|-0.03
TDX005|trades|One Retained-Salary Slot|Clubs may carry one retained contract.|retention_slots=-2|small=0.05,contender=-0.15|-0.05
TDX006|trades|70% Retention Allowed|Clubs may retain up to 70% of a contract.|retention_max_pct=20.0|contender=0.15,rich=0.05|-0.08
TDX007|trades|30% Retention Cap|Salary retention capped at 30%.|retention_max_pct=-20.0|poor=0.05,contender=-0.15|-0.05
TDX008|trades|Single Retention Only|A contract can only be retained once.|retention_max_pct=-5.0,trade_volume=-2.0||0.03
TDX009|trades|Playoff Trade Window|Eliminated clubs may trade during the playoffs.|trade_volume=5.0,fan=0.2|rebuilder=0.1|0.02
TDX010|trades|Unrestricted Draft-Day Trades|No restrictions on trades during the Entry Draft.|trade_volume=4.0,fan=0.3||0.06
TDX011|trades|Offseason-Only Trade Requests|Players may only request trades in the offseason.|trade_demand_rate=-12.0|all=0.06|0.03
TDX012|trades|Public Request Fines|Players fined for public trade demands.|trade_demand_rate=-8.0,fan=-0.2||0.02
TDX013|trades|Veteran Trade Rights|Players with ten seasons can force trades.|trade_demand_rate=12.0,trade_volume=4.0|all=-0.08|-0.08
TDX014|trades|Cap Dump Limit|Clubs may make only two cap-dump trades a season.|trade_volume=-5.0|cap_tight=-0.15,cap_room=0.05|0.0
TDX015|trades|Five-Year Pick Trading|Picks up to five drafts away may be traded.|trade_volume=6.0|contender=0.1,rebuilder=-0.05|0.0
TDX016|trades|Two-Year Pick Trading|Only the next two drafts' picks can be traded.|trade_volume=-6.0|rebuilder=0.1,contender=-0.1|0.0
TDX017|trades|Consecutive First-Rounder Rule|Clubs can't trade first-rounders in consecutive years.|trade_volume=-4.0|rebuilder=0.1|0.04
TDX018|trades|League Trade Veto|The league can veto lopsided trades.|trade_volume=-3.0,fan=0.1||0.0
TDX019|trades|Waiver Claim Limit|Clubs may claim at most five players a season.|trade_volume=-1.0||0.02
TDX020|trades|Waiver Wire Expanded|All demotions of players over 26 go on waivers.|trade_volume=3.0|rebuilder=0.1,contender=-0.05|0.0
TDX021|trades|Conditional Picks Banned|No conditional picks in trades.|trade_volume=-4.0||-0.02
TDX022|trades|Cash-for-Player Trades|Clubs may sell players for cash.|trade_volume=6.0,opex=-0.2|rich=0.15,poor=-0.1|-0.08
TDX023|trades|Draft Rights Trade Freely|Unsigned draft rights trade freely.|trade_volume=3.0||0.05
TDX024|trades|Business-Hours Trade Calls|Trade calls limited to business hours.|trade_volume=-3.0||0.0
TDX025|trades|New Signing Trade Ban|New free-agent signings can't be traded for half a season.|trade_volume=-2.0,trade_demand_rate=-1.0||0.05
TDX026|trades|Immediate Re-Acquisition|Traded players may return to their old club right away.|trade_volume=2.0||0.02
TDX027|trades|Deadline Call-Up Freedom|Clubs may call up players freely after the deadline.|trade_volume=1.0|contender=0.08|0.05
TDX028|trades|Deadline Showcase Week|A week of deadline programming boosts interest.|trade_volume=4.0,fan=0.4,rev_all=0.2||0.08
TDX029|trades|Trade Bonus Clauses Banned|No trade bonuses in contracts.|trade_demand_rate=-2.0||0.02
TDX030|trades|AHL Trades Any Time|AHL players may be traded at any time.|trade_volume=2.0||0.06
TDX031|trades|Retention Costs a Pick|Retaining salary requires a pick sweetener.|retention_max_pct=-5.0,trade_volume=-3.0|cap_tight=-0.1|0.0
TDX032|trades|One Trade Request Per Career|Players may request one trade in their career.|trade_demand_rate=-15.0|all=0.08|0.0
TDX033|trades|Agent Trade Consultation|Agents must be consulted before trades.|trade_volume=-3.0,trade_demand_rate=-3.0||0.0
TDX034|trades|Trade Interest Data Hub|The league shares trade-interest data.|trade_volume=5.0||0.05
TDX035|trades|Holiday Freeze Lifted|Trades allowed through the holidays.|trade_volume=4.0||0.04
DFX001|draft|Draft Age Raised to 19|Players must be 19 to be drafted.|scouting_budget=-5.0|rebuilder=-0.05|-0.02
DFX002|draft|Draft Age Lowered to 17|Late-birthday 17-year-olds can be drafted.|scouting_budget=6.0,opex=0.1|rebuilder=0.05|-0.05
DFX003|draft|Eight-Round Draft|The draft adds an eighth round.|scouting_budget=4.0,opex=0.1|rebuilder=0.1|0.0
DFX004|draft|Five-Round Draft|The draft shrinks to five rounds.|scouting_budget=-6.0,opex=-0.1|rebuilder=-0.1|-0.03
DFX005|draft|Flattened Lottery|Every non-playoff club gets near-equal lottery odds.|fan=0.3|contender=0.05,rebuilder=-0.15|0.0
DFX006|draft|Lottery Win Limit|A club can win the lottery twice in five years.|fan=0.2|rebuilder=-0.05|0.06
DFX007|draft|Gold Plan Draft Order|Draft order set by points after elimination.|fan=0.6|rebuilder=-0.1|0.0
DFX008|draft|Expanded Combine Testing|The combine adds skating and hockey-sense tests.|scouting_budget=6.0,opex=0.1||0.08
DFX009|draft|Optional Combine|Prospects may skip the combine.|scouting_budget=-3.0||0.0
DFX010|draft|European Scouting Office|A league office in Stockholm shares reports.|scouting_budget=8.0,star_rev=1.0||0.06
DFX011|draft|NCAA Practice Access|Scouts get practice access at colleges.|scouting_budget=5.0||0.06
DFX012|draft|Junior Scouting Fee|Clubs pay a fee to scout junior games.|scouting_budget=-4.0,opex=0.1|rich=0.05,poor=-0.1|-0.03
DFX013|draft|Scouting Matching Fund|The league matches small-market scouting spend.|scouting_budget=7.0,revenue_share=0.2|small=0.2,large=-0.05|0.02
DFX014|draft|Scouting Spend Hard Cap|Scouting spend capped league-wide.|scouting_budget=-12.0,opex=-0.2|small=0.15,rich=-0.15|-0.02
DFX015|draft|Prospect Data Pooling|Clubs share tracking data on prospects.|scouting_budget=6.0||0.05
DFX016|draft|Video Scouting Library|The league hosts video of every draft-eligible game.|scouting_budget=7.0,opex=0.1||0.1
DFX017|draft|Prospect Interview Limit|Clubs may interview at most 40 prospects.|scouting_budget=-3.0||0.0
DFX018|draft|Longer Draft Rights|Clubs hold drafted players' rights a year longer.|scouting_budget=2.0|rebuilder=0.1|0.04
DFX019|draft|Unsigned Pick Compensation|Clubs get a compensatory pick for unsigned first-rounders.|scouting_budget=1.0|rebuilder=0.05|0.06
DFX020|draft|Prime-Time First Round|Round one moves to prime time.|rev_all=0.3,fan=0.4||0.1
DFX021|draft|Small-Market Draft Hosts|The draft rotates through small markets.|rev_small=0.4,fan=0.3|small=0.1|0.05
DFX022|draft|Mock Draft Media Rights|The league sells mock-draft media rights.|rev_all=0.2,fan=0.2||0.08
DFX023|draft|Under-18 Showcase|An under-18 showcase hosted by the league.|scouting_budget=4.0,fan=0.3,opex=0.1||0.06
DFX024|draft|Scout Travel Pool|Scouting travel costs pooled.|scouting_budget=3.0,opex=-0.1|small=0.1|0.05
DFX025|draft|Pick Value Chart Published|The league publishes a pick-value chart.|trade_volume=2.0,fan=0.1||0.06
DFX026|draft|Compensation for Lost UFAs|Clubs losing UFAs get extra picks.|trade_demand_rate=-2.0|small=0.15,large=-0.1|0.02
DFX027|draft|Faster Re-Entry|Unsigned picks re-enter the next draft sooner.|scouting_budget=1.0||0.04
DFX028|draft|Shared Combine Medicals|Combine medicals shared with every club.|scouting_budget=3.0,injury_rate=-0.2||0.08
DFX029|draft|Undrafted Signing Window|Undrafted players sign only after July 15.|scouting_budget=1.0||0.02
DFX030|draft|European Combine|The combine alternates to Europe.|scouting_budget=3.0,star_rev=1.0,opex=0.2||0.0
DFX031|draft|Ten-Scout Minimum|Each club employs at least ten scouts.|scouting_budget=6.0,opex=0.3|poor=-0.1|0.0
DFX032|draft|Fan-Voted Pick|Fans vote on one seventh-round pick.|fan=0.4||0.04
DFX033|draft|Junior Rink Tracking|The league installs tracking in junior rinks.|scouting_budget=9.0,opex=0.2||0.04
DFX034|draft|Agent Contact Ban for Minors|Agents barred from prospects under 17.|scouting_budget=1.0,fa_bonus_demand=-1.0||0.05
DFX035|draft|Live Lottery Odds|Lottery odds shown live all season.|fan=0.3||0.08
DEV001|development|AHL Salary Floor Raised|The AHL minimum salary rises.|opex=0.3,min_salary_m=0.01||0.02
DEV002|development|Affiliate Travel Subsidy|The league pays affiliate travel.|opex=-0.2,scouting_budget=2.0|small=0.1|0.06
DEV003|development|Development Coach Minimum|Every club employs two development coaches.|scouting_budget=3.0,opex=0.2||0.05
DEV004|development|AHL Veteran Limit Tightened|Fewer veterans allowed in the AHL.|scouting_budget=2.0,trade_volume=1.0|rebuilder=0.1|0.0
DEV005|development|AHL Veteran Limit Relaxed|More veterans allowed in the AHL.|trade_volume=-1.0|contender=0.05|0.0
DEV006|development|Junior Transfer Rule Eased|19-year-olds may play in the AHL.|scouting_budget=4.0|rebuilder=0.15|-0.03
DEV007|development|Junior Development Payments|Clubs pay junior teams more per pro signing.|opex=0.2,scouting_budget=1.0||0.03
DEV008|development|ECHL Affiliate Mandate|Every club must run an ECHL affiliate.|opex=0.4,scouting_budget=3.0|poor=-0.15|-0.03
DEV009|development|Prospect Camp Funding|The league funds summer development camps.|scouting_budget=3.0,opex=0.1||0.08
DEV010|development|Longer Conditioning Loans|Veterans can be loaned to the AHL for three weeks.|injury_rate=-0.5,trade_volume=1.0||0.04
DEV011|development|Waiver Exemption Shortened|Prospects lose waiver exemption a year sooner.|trade_volume=3.0|rebuilder=-0.1,contender=0.05|-0.03
DEV012|development|European Loans|Prospects may be loaned to European clubs.|scouting_budget=2.0,star_rev=1.0||0.06
DEV013|development|NCAA Eligibility Partnership|Junior players keep NCAA eligibility.|scouting_budget=3.0||0.08
DEV014|development|Shorter Junior Rights|Clubs hold junior draft rights for two years.|scouting_budget=-2.0,trade_volume=1.0|rebuilder=-0.1|-0.02
DEV015|development|Longer Junior Rights|Clubs hold junior draft rights for four years.|scouting_budget=2.0|rebuilder=0.1|0.02
DEV016|development|AHL Playoff Revenue Share|AHL playoff gates shared with parent clubs.|playoff_rev=2.0,rev_all=0.1||0.06
DEV017|development|AHL Streaming Deal|AHL games streamed nationally.|rev_all=0.2,fan=0.2||0.1
DEV018|development|AHL Tracking Data|AHL tracking data feeds parent clubs.|scouting_budget=4.0,opex=0.1||0.06
DEV019|development|Grassroots Hockey Fund|The league invests in minor hockey.|fan=0.5,opex=0.2,expansion_pressure=3.0||0.08
DEV020|development|Girls' Hockey Pipeline|The league funds girls' hockey programs.|fan=0.5,opex=0.1||0.1
DEV021|development|Skills Coach Certification|The league certifies skills coaches.|scouting_budget=2.0||0.08
DEV022|development|Bigger AHL Rosters|AHL rosters grow by two.|opex=0.2,scouting_budget=1.0||0.02
DEV023|development|Post-Deadline Recall Limit|Only four recalls after the deadline.|trade_volume=-1.0|contender=-0.05|0.03
DEV024|development|Unlimited Emergency Recalls|Emergency recalls never count against limits.|injury_rate=-0.3|contender=0.05|0.06
DEV025|development|Homegrown Player Credit|Clubs earn cap credit for homegrown players.|cap_adjust_m=0.3,scouting_budget=2.0|rebuilder=0.15|0.02
DEV026|development|Development Levy|Clubs pay 0.5% of revenue to fund development.|opex=0.3,scouting_budget=3.0|rich=0.05,poor=-0.1|0.0
DEV027|development|Shorter AHL Season|The AHL season trimmed to 68 games.|injury_rate=-0.3,opex=-0.1||0.04
DEV028|development|AHL Goalie Coaches|Each AHL affiliate employs a goalie coach.|scouting_budget=1.0,opex=0.1||0.08
DEV029|development|Prospect Housing Program|The league subsidises young players' housing.|trade_demand_rate=-2.0,opex=0.1||0.06
DEV030|development|AHL Teams in Bid Cities|New AHL teams in expansion bid cities.|expansion_pressure=6.0,rev_all=0.1||0.04
MED001|media|All-Access Series|Every club opens its doors to a season-long show.|fan=0.6,star_rev=3.0,trade_demand_rate=1.0||0.06
MED002|media|Mic'd Up Every Game|One player per team wears a live mic.|fan=0.5,star_rev=2.0||0.06
MED003|media|Locker Rooms Closed to Media|Media banned from locker rooms.|fan=-0.4,trade_demand_rate=-1.0||-0.02
MED004|media|Coaches' Postgame Mandate|Coaches must speak after every game.|fan=0.2||0.08
MED005|media|Game-Day Social Blackout|No player posts on game days.|fan=-0.4,star_rev=-2.0||-0.03
MED006|media|Creator Partnerships|Clubs pay creators to cover games.|fan=0.4,rev_all=0.2,opex=0.1||0.06
MED007|media|Alternate Broadcasts|Kids and analytics broadcasts every week.|fan=0.5,rev_all=0.3||0.08
MED008|media|Odds on Broadcasts|Live betting odds shown on national TV.|rev_all=0.6,fan=-0.5||0.0
MED009|media|Bench Interviews|Players interviewed on the bench during games.|fan=0.3,star_rev=1.0||0.02
MED010|media|Highlights Freed|Fans may share highlights freely.|fan=0.7,star_rev=2.0,rev_all=-0.2||0.06
MED011|media|Highlights Locked Down|Highlights only on league platforms.|rev_all=0.3,fan=-0.5||-0.02
MED012|media|Fan Council|A fan council advises the league.|fan=0.5||0.1
MED013|media|Stoppage Music Freedom|Clubs play music freely during stoppages.|fan=0.2,home_ice=3.0||0.06
MED014|media|Standard Goal Celebrations|Uniform in-arena goal celebrations.|fan=-0.3||-0.05
MED015|media|Saturday Doubleheaders|National Saturday doubleheaders.|rev_all=0.4,fan=0.3||0.08
MED016|media|Christmas Day Games|Two marquee games on Christmas Day.|rev_all=0.5,fan=0.4,injury_rate=0.2||0.02
MED017|media|Thanksgiving Showcase|A U.S. Thanksgiving tripleheader.|rev_all=0.4,fan=0.3||0.06
MED018|media|Star Marketing Campaign|League ads built around ten stars.|star_rev=6.0,fan=0.3||0.08
MED019|media|Rookie Marketing Campaign|The league promotes its rookies.|star_rev=3.0,fan=0.3|rebuilder=0.05|0.08
MED020|media|Legends Night|Each club honours its legends every season.|fan=0.4,rev_all=0.1||0.1
MED021|media|All-Star Fan Fest|A bigger All-Star fan festival.|rev_all=0.3,fan=0.4||0.08
MED022|media|All-Star Game Dropped|The All-Star Game is cancelled for rest.|injury_rate=-0.5,rev_all=-0.3,fan=-0.3||-0.02
MED023|media|All-Star Skills Expansion|More skills events and prize money.|star_rev=2.0,fan=0.4,opex=0.1||0.06
MED024|media|Player Podcasts|Players may host podcasts in season.|star_rev=2.0,trade_demand_rate=1.0||0.02
MED025|media|Broadcaster Neutrality|National broadcasters can't work for clubs.|fan=0.2||0.06
MED026|media|Arena Tours and Museums|Clubs open arena tours year-round.|rev_all=0.2,fan=0.2||0.1
MED027|media|Video Game Revenue Share|Clubs share video game licensing.|rev_all=0.4,star_rev=1.0||0.12
MED028|media|Official Fantasy League|The league runs official fantasy hockey.|fan=0.4,star_rev=2.0,rev_all=0.2||0.1
MED029|media|Live Anthems|Live anthem performances at every game.|fan=0.2,opex=0.05||0.1
MED030|media|Fewer In-Arena Ads|In-arena advertising cut back.|fan=0.5,rev_all=-0.3||0.0
MED031|media|Augmented Reality Broadcasts|AR stat overlays on TV.|fan=0.3,rev_all=0.2,opex=0.1||0.08
MED032|media|Spanish-Language Broadcasts|Every national game also in Spanish.|fan=0.3,rev_all=0.2,expansion_pressure=2.0||0.08
MED033|media|Custom Warmup Gear|Players may wear custom gear in warmups.|star_rev=2.0,fan=0.3||0.06
MED034|media|Arena Wi-Fi Mandate|Free Wi-Fi in every arena.|fan=0.4,opex=0.2||0.06
MED035|media|Club Preview Specials|Each club gets a national season-preview show.|fan=0.3,rev_all=0.2||0.1
ARN001|arena|16,000-Seat Minimum|New arenas must seat at least 16,000.|value_growth=0.3,opex=0.3,relocation_ease=0.02|large=0.1,poor=-0.15|-0.03
ARN002|arena|Standard Lease Terms|League standard lease terms with host cities.|opex=-0.4,relocation_ease=-0.02|poor=0.15|0.05
ARN003|arena|Public Arena Funding Push|The league lobbies for public arena money.|value_growth=0.5,relocation_ease=0.02|poor=0.1,rich=0.05|0.0
ARN004|arena|Green Arena Standard|Arenas cut energy use by 30%.|opex=0.3,fan=0.4||0.04
ARN005|arena|Arena Upgrade Loan Fund|The league lends for upgrades at low rates.|value_growth=0.4,opex=-0.2,relocation_ease=-0.02|poor=0.15|0.04
ARN006|arena|Arena Security Standards|Uniform screening at every arena.|opex=0.2,fan=-0.1||0.08
ARN007|arena|Transit Access Requirement|New arenas must be on transit lines.|fan=0.3,value_growth=0.2,opex=0.1||0.04
ARN008|arena|Arena District Rights|Clubs may develop the land around their arenas.|value_growth=0.8,rev_all=0.3|rich=0.1|0.04
ARN009|arena|Sale-Leaseback Allowed|Owners may sell arenas and lease them back.|value_growth=-0.2,opex=0.2|poor=0.1|0.0
ARN010|arena|Monthly Ice Inspection|League inspectors grade the ice every month.|opex=0.1,injury_rate=-0.3,scoring_pace=0.5||0.08
ARN011|arena|Exact Rink Dimensions|Every rink exactly 200 by 85 feet.|home_ice=-6.0,opex=0.2||0.04
ARN012|arena|Seamless Glass Returns|Flexible glass replaced with seamless panels.|injury_rate=0.3,fan=0.2||0.0
ARN013|arena|Accessibility Upgrade|More accessible seating in every arena.|fan=0.4,opex=0.2||0.08
ARN014|arena|Shared Arena Agreements|Clubs may share arenas with basketball tenants.|opex=-0.3,fan=-0.1||0.02
ARN015|arena|Arena Name Approval|The league approves arena names.|rev_all=-0.1,fan=0.1||0.0
ARN016|arena|Wider Concourses|Wider concourses in every arena.|rev_all=0.3,opex=0.2||0.02
ARN017|arena|19,000-Seat Cap|Arenas capped at 19,000 seats.|rev_large=-0.4,fan=0.1|small=0.05,large=-0.1|-0.02
ARN018|arena|Practice Rink Minimum|Every club runs its own practice rink.|opex=0.3,injury_rate=-0.3|poor=-0.1|0.0
ARN019|arena|Parking Revenue Sharing|Arena parking revenue shared with visitors.|revenue_share=0.2||0.0
ARN020|arena|Arena Age Limit|Arenas over 40 years old must be replaced.|value_growth=0.4,opex=0.4,relocation_ease=0.04|rich=0.05,poor=-0.2|-0.08
ARN021|arena|Downtown Arena Incentive|A league bonus for downtown arenas.|value_growth=0.3,fan=0.2||0.05
ARN022|arena|Southern Arena Cooling|Cooling upgrades in warm-weather arenas.|opex=0.2,scoring_pace=0.3||0.04
ARN023|arena|Arena Control Requirement|Clubs must own or control their arena.|value_growth=0.4,relocation_ease=-0.04|rich=0.1,poor=-0.1|-0.03
ARN024|arena|Arena Bond Program|League-backed bonds for arena builds.|value_growth=0.5,relocation_ease=-0.03|poor=0.1|0.04
ARN025|arena|Visitor Room Standard|Visiting dressing rooms must match the home room.|home_ice=-8.0,opex=0.1||0.06
INT001|international|European Division Study|The league studies a European division.|expansion_pressure=8.0,star_rev=2.0||0.0
INT002|international|Global Series Expansion|Ten regular-season games abroad each year.|rev_all=0.6,star_rev=3.0,injury_rate=0.5|poor=-0.05|0.02
INT003|international|Olympic Participation Ended|No more Olympic breaks.|injury_rate=-1.0,star_rev=-4.0,fan=-0.6|all=-0.03|-0.05
INT004|international|World Championship Release|Players freed for the World Championship.|star_rev=2.0,injury_rate=0.5||0.04
INT005|international|Higher Transfer Fees|Clubs pay more to sign European players.|opex=0.2,scouting_budget=-2.0||-0.02
INT006|international|European Agreement Renewed|The transfer agreement with European leagues is renewed.|scouting_budget=3.0||0.1
INT007|international|Asian Exhibition Tour|Preseason games in Asia.|star_rev=2.0,rev_all=0.3,injury_rate=0.3||0.04
INT008|international|Melbourne Showcase|A preseason series in Australia.|star_rev=1.0,rev_all=0.2||0.04
INT009|international|Lottery Show Abroad|The draft lottery held in a European city.|fan=0.2,star_rev=1.0||0.05
INT010|international|Visa Support Program|The league handles work visas for imports.|opex=0.1,scouting_budget=2.0||0.1
INT011|international|Best-on-Best Revenue Split|Best-on-best tournament revenue shared with clubs.|rev_all=0.5,injury_rate=0.3||0.08
INT012|international|Overseas Broadcast Hubs|League studios in London and Stockholm.|rev_all=0.4,star_rev=2.0,opex=0.2||0.04
INT013|international|Mexico City Games|Neutral-site games in Mexico City.|rev_all=0.2,expansion_pressure=3.0||0.04
INT014|international|Import Player Limit|Clubs limited to eight European players.|scouting_budget=-4.0,star_rev=-2.0|small=0.05|-0.08
INT015|international|Global Fan Membership|Paid international fan clubs.|rev_all=0.3,star_rev=1.0||0.08
INT016|international|Russian Transfer Agreement|A formal deal for players moving from Russia.|scouting_budget=3.0||0.04
INT017|international|World Juniors Release Guarantee|Clubs must release players for the World Juniors.|fan=0.4,injury_rate=0.2|contender=-0.05|0.04
INT018|international|Partner Arenas Abroad|Partner arenas overseas host showcases.|rev_all=0.3,star_rev=1.0||0.04
INT019|international|Cross-Border Tax Equalisation|The league offsets tax differences for players.|opex=0.3,trade_demand_rate=-2.0|small=0.05|0.0
INT020|international|Global Youth Clinics|Stars run clinics overseas every summer.|star_rev=2.0,fan=0.3,injury_rate=0.1||0.06
STX001|structure|Four-Fifths Relocation Vote|Relocations need four-fifths of the Board.|relocation_ease=-0.06|small=0.1,poor=-0.1|0.02
STX002|structure|Relocation Fast Track|Clubs losing $20M+ may apply to move after one season.|relocation_ease=0.06|poor=0.2,small=0.05|-0.05
STX003|structure|Expansion Market Study|The league ranks every bid city.|expansion_pressure=12.0||0.05
STX004|structure|Expansion Fee Floor|Expansion fees can't fall below twice the average club value.|expansion_pressure=-8.0,value_growth=0.3|rich=0.1|0.02
STX005|structure|Contraction Review Clause|The league may buy out failing clubs.|relocation_ease=0.04,expansion_pressure=-10.0|rich=0.05,poor=-0.2|-0.1
STX006|structure|Geographic Realignment|Divisions redrawn to cut travel.|opex=-0.4,fan=0.2||0.05
STX007|structure|Rivalry-First Schedule|More division games, fewer cross-conference trips.|rev_all=0.3,fan=0.4,opex=-0.2||0.05
STX008|structure|Balanced Schedule|Every team plays every other team equally.|fan=-0.2,opex=0.2||-0.02
STX009|structure|Ownership Transparency|Owners disclose their finances to the Board.|relocation_ease=-0.02,fan=0.2||0.02
STX010|structure|Minority Stake Sales|Owners may sell stakes of up to 30%.|value_growth=0.6||0.08
STX011|structure|Ownership Group Limit|No more than 25 partners per club.|value_growth=-0.2||0.0
STX012|structure|Old-City Compensation|Moving clubs fund a minor-league team in the old city.|relocation_ease=-0.04,opex=0.1|rich=0.05,poor=-0.1|0.02
STX013|structure|Expansion Waiting Period|New expansion bids wait three seasons after the last one.|expansion_pressure=-6.0||0.04
STX014|structure|Canadian Bids First|Canadian expansion bids get the first look.|expansion_pressure=5.0|small=0.05|0.0
STX015|structure|Commissioner Term Limits|The commissioner is limited to two terms.|fan=0.3||0.02
"""
