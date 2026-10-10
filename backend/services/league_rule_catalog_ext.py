"""
500 more Board of Governors proposals (on top of the 200 in league_rule_catalog).

At five ballots a meeting, 700 rules is 140 seasons before the Board would ever have
to repeat one — and the drawer never repeats a rule until every rule has had a vote.

Many of these are on-ice rules that change how games are played (scoring pace,
penalties, power plays, home ice, overtime length, shootout vs ties, points system);
those keys are read by app.sim_engine.league.game_rules. Every other effect key is
consumed by the same live systems as the base catalog (see league_governance.EFFECT_SPECS).

Row format (one rule per line):
    ID|category|Title|Summary|effect=value,...|trait=lean,...|base
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from services.league_rule_catalog_ext_data import DATA_B

EXTRA_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "onice": {"label": "On-Ice Rules", "threshold": "majority"},
    "officiating": {"label": "Officiating & Discipline", "threshold": "majority"},
    "health": {"label": "Player Health & Safety", "threshold": "majority"},
    "development": {"label": "Minor Leagues & Development", "threshold": "two_thirds"},
    "media": {"label": "Media & Fan Experience", "threshold": "majority"},
    "arena": {"label": "Arenas & Operations", "threshold": "two_thirds"},
    "international": {"label": "International & Events", "threshold": "majority"},
}

DATA_A = """
ICE001|onice|Four-Minute Overtime|Overtime trimmed to four minutes: more shootouts.|ot_minutes=4,fan=-0.2||0.0
ICE002|onice|Six-Minute Overtime|Overtime runs six minutes of 3-on-3.|ot_minutes=6,fan=0.3||0.05
ICE003|onice|Seven-Minute Overtime|Seven minutes of 3-on-3 before a shootout.|ot_minutes=7,fan=0.4,injury_rate=0.3||0.04
ICE004|onice|Eight-Minute Overtime|Eight-minute overtime settles more games in play.|ot_minutes=8,fan=0.5,injury_rate=0.4||0.03
ICE005|onice|Fifteen-Minute Overtime|A full 15-minute overtime before any shootout.|ot_minutes=15,fan=0.6,injury_rate=1.0|contender=0.05|-0.05
ICE006|onice|Twenty-Minute Sudden Death|A full 20-minute sudden-death period in the regular season.|ot_minutes=20,fan=0.4,injury_rate=1.5|all=-0.05|-0.1
ICE007|onice|Ties Return After Five Minutes|Shootout scrapped; a tie after five minutes of OT stands.|ties_allowed=1,ot_minutes=5,fan=-0.6||-0.05
ICE008|onice|Ties After Ten-Minute Overtime|Ten minutes of overtime, then the game ends tied.|ties_allowed=1,ot_minutes=10,fan=-0.2||0.0
ICE009|onice|Shootout Restored|The shootout returns to decide regular-season ties.|ties_allowed=0,fan=0.5||0.08
ICE010|onice|3-2-1-0 Points System|Three for a regulation win, two for an OT win, one for an OT loss.|pts_reg_win=3,pts_ot_win=2,pts_otl=1,fan=0.4|contender=0.05|0.02
ICE011|onice|No Loser Point|Overtime and shootout losses earn nothing.|pts_otl=0,fan=0.3|contender=0.1,rebuilder=-0.1|-0.05
ICE012|onice|Loser Point Restored|Overtime and shootout losses earn a point again.|pts_otl=1,fan=0.1|rebuilder=0.1|0.05
ICE013|onice|Regulation Premium, No Loser Point|Three for a regulation win, two for any extra-time win, nothing for a loss.|pts_reg_win=3,pts_ot_win=2,pts_otl=0,fan=0.3|contender=0.1|-0.05
ICE014|onice|Classic Two-Point Win|Back to two points for every win and one for an OT loss.|pts_reg_win=2,pts_ot_win=2,pts_otl=1||0.05
ICE015|onice|League-Wide Shot Clock|35-second offensive-zone shot clock in every arena.|scoring_pace=5.0,fan=-0.2|all=-0.04|-0.05
ICE016|onice|Two-Line Pass Returns|The two-line pass is illegal again: fewer stretch passes.|scoring_pace=-3.0,fan=-0.4||-0.08
ICE017|onice|Hurry-Up Faceoff Rule|Linesmen drop the puck faster after whistles.|scoring_pace=1.0,fan=0.3||0.1
ICE018|onice|Larger Nets Trial|Nets two inches wider for one season.|scoring_pace=7.0,fan=0.2|all=-0.06|-0.12
ICE019|onice|Goalie Trapezoid Removed|Goalies may play the puck anywhere behind the goal line.|scoring_pace=-1.5||0.02
ICE020|onice|Trapezoid Narrowed|Smaller area for goalies to handle the puck.|scoring_pace=1.0,fan=0.1||0.05
ICE021|onice|Offside Review Eliminated|Offside can no longer be challenged.|scoring_pace=1.5,fan=0.4||0.08
ICE022|onice|Goalie Glove Size Cut|Catching gloves shrink by two inches.|scoring_pace=3.0,fan=0.3||0.05
ICE023|onice|Contoured Chest Protectors|Goalie chest protectors must follow the body.|scoring_pace=2.0,injury_rate=0.3||0.04
ICE024|onice|Delayed Penalty Ends on Any Touch|A delayed penalty ends the moment the offending team touches the puck.|pp_efficiency=-2.0,scoring_pace=-0.5||0.0
ICE025|onice|Minors Served in Full|Power plays don't end when a goal is scored.|pp_efficiency=10.0,fan=0.3|all=-0.04|-0.04
ICE026|onice|No Icing While Shorthanded|Shorthanded teams can no longer ice the puck freely.|pp_efficiency=8.0,fan=0.3||-0.03
ICE027|onice|Four-on-Four Coincidental Minors|Coincidental minors are played 4-on-4.|scoring_pace=1.0,fan=0.2||0.08
ICE028|onice|Double Minors Abolished|Double minors become single minors.|pp_efficiency=-3.0||0.0
ICE029|onice|Neutral-Zone Draw After Icing|No more defensive-zone draws after icing.|scoring_pace=-1.0||0.02
ICE030|onice|Penalty-Kill Icing Called|Shorthanded icing is called like even strength.|pp_efficiency=7.0||-0.02
ICE031|onice|Hand-Pass Goals Allowed|Goals directed by a hand pass count.|scoring_pace=1.0,fan=-0.1||-0.03
ICE032|onice|Kicked-In Goals Allowed|Any goal off a skate counts unless kicked violently.|scoring_pace=1.5,fan=0.2||0.04
ICE033|onice|High-Stick Line Raised|Goals off sticks up to the crossbar count.|scoring_pace=1.0||0.03
ICE034|onice|Crease Rule Tightened|Any attacker in the crease waves off a goal.|scoring_pace=-2.0,fan=-0.3||-0.05
ICE035|onice|Crease Contact Relaxed|Incidental crease contact no longer cancels goals.|scoring_pace=2.0,injury_rate=0.5,fan=0.2||0.0
ICE036|onice|Home Last Change Removed|Both benches change at the same time.|home_ice=-25.0||-0.05
ICE037|onice|Home Matchup Advantage Expanded|Home coaches also get the last change after icings.|home_ice=15.0,fan=0.2|large=0.05|0.0
ICE038|onice|Visitor Bench Standards|Visiting bench conditions standardised league-wide.|home_ice=-10.0,opex=0.1||0.08
ICE039|onice|Crowd Noise Limits|Arena PA and noise limits during play.|home_ice=-8.0,fan=-0.3||-0.05
ICE040|onice|Board and Glass Uniformity|Uniform boards and glass in every arena.|home_ice=-12.0,opex=0.3,injury_rate=-0.5||0.02
ICE041|onice|Altitude Travel Allowance|Visitors get an extra day before high-altitude games.|home_ice=-6.0,opex=0.2||0.0
ICE042|onice|Goalie Pull Restriction|Goalies can't be pulled before the final two minutes.|scoring_pace=-1.0||-0.02
ICE043|onice|Tag-Up Zone Expanded|Players can tag up anywhere along the blue line.|scoring_pace=1.0,fan=0.1||0.06
ICE044|onice|Changes on the Fly Only|No line changes after icings for either team.|scoring_pace=1.5,injury_rate=0.5||0.0
ICE045|onice|Faceoff Encroachment Crackdown|Wingers encroaching on draws cost the faceoff.|penalty_rate=2.0||0.02
ICE046|onice|Skater Puck-Freeze Minor|Skaters closing a hand on the puck draw a minor.|penalty_rate=3.0,pp_efficiency=1.0||0.0
ICE047|onice|Goalie Freeze Limit|Goalies freezing pucks outside the crease get a minor.|penalty_rate=2.0,scoring_pace=1.0||0.0
ICE048|onice|Net Dislodgement Relaxed|Accidental net dislodging becomes a faceoff.|penalty_rate=-2.0||0.05
ICE049|onice|Overtime Opens 4-on-4|Eight minutes of overtime: 4-on-4 first, then 3-on-3.|ot_minutes=8,scoring_pace=-0.5,fan=0.3||0.0
ICE050|onice|Overtime Possession Rule|Teams can't regroup into their own zone in overtime.|scoring_pace=0.5,fan=0.4||0.05
ICE051|onice|Five-Round Shootout|Shootouts run five shooters instead of three.|fan=0.4,star_rev=1.0||0.08
ICE052|onice|Penalty Shot for Breakaway Fouls|Any foul on a clear breakaway is a penalty shot.|scoring_pace=0.5,fan=0.3||0.06
ICE053|onice|Diving Becomes a Double Minor|Embellishment draws four minutes.|penalty_rate=2.0,fan=0.2||0.0
ICE054|onice|Slashing Standard Enforced|Any slash on the hands is a minor.|penalty_rate=6.0,pp_efficiency=1.0,injury_rate=-1.0||0.04
ICE055|onice|Obstruction Crackdown|Hooking and holding called to the letter for a season.|penalty_rate=12.0,scoring_pace=2.0,fan=-0.2|all=-0.05|-0.05
ICE056|onice|Let Them Play Directive|Officials call only clear infractions.|penalty_rate=-10.0,scoring_pace=-1.5,injury_rate=1.0||0.0
ICE057|onice|Net-Front Cross-Check Standard|Cross-checks in front of the net are minors.|penalty_rate=5.0,injury_rate=-1.0||0.03
ICE058|onice|Interference Called Everywhere|Picks away from the puck are called all over the ice.|penalty_rate=4.0,scoring_pace=1.0||0.02
ICE059|onice|Five-on-Three Limit|Two-man advantages capped at 60 seconds.|pp_efficiency=-4.0||0.0
ICE060|onice|Power Plays Start in the Zone|Every power play opens with an offensive-zone draw.|pp_efficiency=4.0||0.02
ICE061|onice|Shorthanded Goal Ends the Penalty|A shorthanded goal releases the penalized player.|pp_efficiency=-1.0,fan=0.2||0.03
ICE062|onice|Monthly Goalie Equipment Audit|Equipment checks every month instead of every season.|scoring_pace=2.0,opex=0.1||0.08
ICE063|onice|Composite Puck Trial|Livelier composite pucks for a season.|scoring_pace=3.0,injury_rate=0.5||-0.05
ICE064|onice|Ice Temperature Standard|Colder, faster ice in every rink.|scoring_pace=1.5,opex=0.3,injury_rate=0.3||0.03
ICE065|onice|Mid-Period Ice Scrape|Ice scraped at the midpoint of every period.|scoring_pace=1.0,opex=0.2,rev_all=0.2||0.02
ICE066|onice|Four-Quarter Game Trial|Four 15-minute quarters with more ad breaks.|rev_all=0.8,fan=-0.8,injury_rate=0.5|poor=0.1|-0.1
ICE067|onice|Longer Intermissions|Intermissions grow to 20 minutes for concessions.|rev_all=0.5,fan=-0.4||0.02
ICE068|onice|Fewer TV Timeouts|Two TV timeouts a period instead of three.|rev_all=-0.4,fan=0.5,injury_rate=0.3||0.0
ICE069|onice|High-Stick Goal Review|Every possible high-stick goal is reviewed.|scoring_pace=-0.5,fan=0.2||0.05
ICE070|onice|Major Penalty Video Review|Referees review every major on video.|penalty_rate=-2.0,pp_efficiency=-1.0,fan=0.3||0.1
ICE071|onice|Roaming Goalies Allowed|Goalies may join plays outside the crease.|scoring_pace=0.5,injury_rate=0.5||-0.02
ICE072|onice|No Overtime Timeouts|Timeouts are banned in overtime.|scoring_pace=0.5,fan=0.2||0.05
ICE073|onice|No Extra Attacker on the Power Play|Teams on a power play may not pull the goalie.|pp_efficiency=-1.0||0.0
ICE074|onice|Too-Many-Men Review|Too-many-men calls can be reviewed.|penalty_rate=-1.0,fan=0.1||0.06
ICE075|onice|Penalty Box Exit Rule|Released players must touch the bench before playing.|pp_efficiency=1.0||0.0
ICE076|onice|Ties Count in the Standings|Ties stand and earn one point each.|ties_allowed=1,fan=-0.3|rebuilder=0.05|-0.08
ICE077|onice|Win-or-Nothing Standings|Two points for any win, nothing for any loss.|pts_reg_win=2,pts_ot_win=2,pts_otl=0,fan=0.2|contender=0.08,rebuilder=-0.08|-0.05
ICE078|onice|Four-Point Regulation Win|Four for a regulation win, two for an OT win, one for an OT loss.|pts_reg_win=4,pts_ot_win=2,pts_otl=1,fan=0.3|contender=0.1|-0.12
ICE079|onice|Neutral Benches in Rivalry Week|Rivalry games use neutral bench assignments.|home_ice=-5.0,fan=0.3||0.02
ICE080|onice|Five-Minute Overtime Guarantee|Overtime fixed at five minutes of 3-on-3 by bylaw.|ot_minutes=5,fan=0.1||0.1
REF001|officiating|Full-Time Officials|Referees paid year-round for better consistency.|opex=0.2,fan=0.2,penalty_rate=-1.0||0.08
REF002|officiating|Referee Accountability Reports|Officials' errors published weekly.|fan=0.4,penalty_rate=-2.0||0.05
REF003|officiating|Third On-Ice Referee Trial|Three referees on the ice in select games.|penalty_rate=6.0,opex=0.3,fan=-0.1||-0.04
REF004|officiating|Rookie Referee Program|More young officials promoted quickly.|penalty_rate=3.0,opex=-0.1||0.0
REF005|officiating|Veteran Referee Retention Bonus|Experienced referees paid to stay.|penalty_rate=-2.0,opex=0.2||0.05
REF006|officiating|Player Safety Department Expanded|More staff reviewing hits.|injury_rate=-1.5,opex=0.2,fan=0.2||0.08
REF007|officiating|Dangerous-Play Fines Doubled|Fines for dangerous plays double.|injury_rate=-1.5,penalty_rate=-1.0||0.06
REF008|officiating|Repeat Offender Escalator|Repeat offenders face doubled suspensions.|injury_rate=-2.0,fan=0.3||0.08
REF009|officiating|Instigator Rule Repealed|Instigators no longer get an extra minor.|penalty_rate=-1.5,injury_rate=1.0,fan=0.3|all=-0.03|-0.04
REF010|officiating|Instigating Becomes a Major|Instigating a fight draws five minutes.|penalty_rate=2.0,injury_rate=-1.0,fan=-0.2||0.0
REF011|officiating|Clutch-and-Grab Memo|Officials told to call all obstruction from puck drop.|penalty_rate=8.0,scoring_pace=1.5||0.0
REF012|officiating|Playoff-Style Officiating All Season|Fewer calls in the regular season.|penalty_rate=-8.0,scoring_pace=-1.0,injury_rate=0.8||-0.03
REF013|officiating|Call Consistency Audit|The league audits call rates every month.|penalty_rate=-1.0,fan=0.3||0.1
REF014|officiating|Head Contact Is a Major|All head contact draws a major.|penalty_rate=4.0,pp_efficiency=2.0,injury_rate=-2.5||0.04
REF015|officiating|Boarding Standard Clarified|A clear standard for hits from behind.|injury_rate=-1.5,penalty_rate=1.0||0.08
REF016|officiating|Charging Defined by Strides|Charging judged by strides taken before contact.|penalty_rate=2.0,injury_rate=-1.0||0.02
REF017|officiating|Kneeing Suspension Minimum|Kneeing carries a one-game minimum suspension.|injury_rate=-1.0||0.05
REF018|officiating|Live Referee Microphones|Referee explanations aired in the arena and on TV.|fan=0.6,rev_all=0.2||0.08
REF019|officiating|Situation Room Final Say|Toronto decides every review.|fan=0.2,penalty_rate=-0.5||0.05
REF020|officiating|Referees Review Their Own Calls|Referees can review their own penalty calls.|penalty_rate=-3.0,fan=0.2||0.04
REF021|officiating|Failed Challenge Double Minor|A failed coach's challenge costs a double minor.|penalty_rate=1.0||0.0
REF022|officiating|Coach's Challenge Abolished|No coach's challenges at all.|scoring_pace=1.0,fan=0.3||0.0
REF023|officiating|Hand-Pass Review Added|Hand passes before goals can be reviewed.|scoring_pace=-0.5||0.02
REF024|officiating|Abuse of Officials Crackdown|Bench minors for verbal abuse double.|penalty_rate=2.0,fan=0.1||0.03
REF025|officiating|Referee Rules on Goalie Interference|Only the referee rules on goalie interference.|scoring_pace=0.5,fan=-0.1||0.0
REF026|officiating|Taunting Minor|Taunting draws a minor.|penalty_rate=1.5,fan=-0.2||-0.02
REF027|officiating|Three-Fight Suspension|A third fight in a season triggers a suspension.|injury_rate=-1.0,fan=-0.3||0.0
REF028|officiating|Fighting Eliminated|Any fight is an automatic one-game suspension.|injury_rate=-3.0,penalty_rate=2.0,fan=-1.0|all=-0.06|-0.1
REF029|officiating|Linesmen Call Minors|Linesmen may call any minor penalty.|penalty_rate=5.0||-0.02
REF030|officiating|Referee Rest Rotation|Referees capped at three games a week.|penalty_rate=-1.0,opex=0.1||0.05
REF031|officiating|Player Safety Video Library|Every ruling explained in a public video.|fan=0.4,injury_rate=-0.5||0.12
REF032|officiating|Hit-Tracking Helmet Sensors|Helmet sensors flag dangerous hits for review.|injury_rate=-2.0,opex=0.3||0.0
REF033|officiating|Late-Hit Window|Hits more than a second late are minors.|penalty_rate=3.0,injury_rate=-1.5||0.03
REF034|officiating|Spearing Major Mandatory|All spearing draws a major.|penalty_rate=1.0,injury_rate=-0.5||0.05
REF035|officiating|Delay-of-Game Review|Puck-over-glass calls can be reviewed.|penalty_rate=-1.0,fan=0.1||0.04
REF036|officiating|Puck-Over-Glass Minor Restored|Every puck over the glass is a minor again.|penalty_rate=3.0||0.0
REF037|officiating|Major Downgrade Review|Majors can be downgraded to minors on review.|pp_efficiency=-2.0,penalty_rate=-1.0||0.06
REF038|officiating|Embellishment Public List|Players fined for diving are named publicly.|penalty_rate=-2.0,fan=0.3||0.05
REF039|officiating|Officials Development Program|The league trains referees from the junior ranks.|opex=0.1,fan=0.1||0.1
REF040|officiating|Blue-Line Cameras|Blue-line cameras assist offside calls.|scoring_pace=0.5,opex=0.2||0.06
REF041|officiating|Goal-Line Puck Tracking|Chip-tracked pucks confirm goals instantly.|fan=0.3,opex=0.2||0.1
REF042|officiating|Penalty-Kill Faceoff Choice|Shorthanded teams pick the faceoff side.|pp_efficiency=-1.5||0.02
REF043|officiating|First Retaliation Waived|A first retaliation goes uncalled.|penalty_rate=-2.0,injury_rate=0.5||-0.03
REF044|officiating|Stick Holding on Chances|Holding the stick on a scoring chance is a penalty shot.|scoring_pace=0.5,penalty_rate=1.0||0.0
REF045|officiating|Diving Suspension Threshold|A fourth diving fine is a one-game ban.|penalty_rate=-1.5,star_rev=-0.5||0.02
REF046|officiating|Penalty Minute Pay Fines|Players forfeit pay for excessive penalty minutes.|penalty_rate=-3.0,trade_demand_rate=2.0||0.0
REF047|officiating|Referee Grades Published|Season grades for every official made public.|fan=0.3,penalty_rate=-0.5||0.05
REF048|officiating|Neutral-Zone Hooking Standard|Hooking in the neutral zone called tighter.|penalty_rate=3.0,scoring_pace=0.5||0.02
REF049|officiating|Bench-Clearing Fines|Clubs fined heavily when benches clear.|injury_rate=-0.5,opex=-0.1||0.08
REF050|officiating|Faceoff Violation Leniency|Faceoff violations no longer become minors.|penalty_rate=-2.0||0.04
HLT001|health|Independent Concussion Doctors|Independent doctors decide return-to-play.|injury_rate=-2.0,fan=0.3,opex=0.2||0.12
HLT002|health|Full Shields for Under-23s|Young players wear full face shields.|injury_rate=-1.5,fan=-0.2||0.03
HLT003|health|Wrist Guard Requirement|Cut-resistant wrist guards for every skater.|injury_rate=-1.0||0.1
HLT004|health|Blade Guard Standard|Blade covers required in warmups and practice.|injury_rate=-0.5||0.12
HLT005|health|Sleep Science Scheduling|Late-night arrivals limited by the schedule.|injury_rate=-1.5,opex=0.2||0.05
HLT006|health|Team Doctor Minimum|Each club staffs two team doctors.|injury_rate=-1.5,opex=0.4|poor=-0.1|0.0
HLT007|health|League Sports Science Center|A central lab tracks player load.|injury_rate=-2.0,opex=0.3||0.05
HLT008|health|Second Opinion Rights|Players choose their own surgeon.|injury_rate=-0.5,trade_demand_rate=-2.0||0.06
HLT009|health|Mental Health Days|Players can take two mental-health days a season.|trade_demand_rate=-4.0,fan=0.3,opex=0.1||0.08
HLT010|health|Nutrition Program Funding|The league pays for team nutritionists.|injury_rate=-1.0,opex=0.2||0.06
HLT011|health|Altitude Recovery Rules|Extra rest after high-altitude trips.|injury_rate=-0.5||0.04
HLT012|health|Bone Density Screening|Annual screening for every player.|injury_rate=-0.5,opex=0.1||0.08
HLT013|health|Arena Heat Standards|Arena temperature caps in warm markets.|injury_rate=-0.5,opex=0.2||0.05
HLT014|health|Skate Fitting Audits|Skate fittings audited by trainers.|injury_rate=-0.3||0.06
HLT015|health|Injury Disclosure Rule|Clubs must disclose injury types.|fan=0.4,trade_volume=2.0|contender=-0.05|0.0
HLT016|health|Vague Injury Reports Allowed|Clubs may list only upper or lower body.|fan=-0.2,trade_volume=-1.0||0.05
HLT017|health|Long-Term IR Audits|LTIR relief audited and capped at the player's cap hit.|cap_adjust_m=0.25,trade_volume=-2.0|cap_tight=-0.15|0.0
HLT018|health|Conditioning Stints|Injured players may play two AHL games to recover.|injury_rate=-0.5||0.08
HLT019|health|Practice Contact Limits|Full-contact practices capped weekly.|injury_rate=-1.5|contender=0.05|0.05
HLT020|health|Shot-Blocking Pad Standard|Mandatory padding for shot blockers.|injury_rate=-1.0,scoring_pace=0.5||0.06
HLT021|health|Neck Guards for Officials|Officials wear cut-resistant neck guards.|opex=0.05,fan=0.1||0.15
HLT022|health|Cardiac Screening|Heart screening at every training camp.|opex=0.1,fan=0.2||0.15
HLT023|health|Spotter in Every Arena|Independent concussion spotters at every game.|injury_rate=-1.0,opex=0.1||0.1
HLT024|health|Mandatory Cool-Down|Players cycle-cool after every game.|injury_rate=-0.5||0.08
HLT025|health|Long-Overtime Rest Rule|No back-to-back after a game with a long overtime.|injury_rate=-0.5,opex=0.1||0.03
HLT026|health|Trainer Certification|All trainers league-certified.|injury_rate=-0.5,opex=0.1||0.08
HLT027|health|Painkiller Oversight|The league monitors painkiller prescriptions.|fan=0.3,injury_rate=0.3||0.06
HLT028|health|Retired Player Health Fund|A league fund for retired players' health.|opex=0.3,fan=0.4||0.08
HLT029|health|Summer Training Logs|Players log summer training with their clubs.|injury_rate=-0.5,trade_demand_rate=2.0||0.0
HLT030|health|Early Return Bonuses|Players paid a bonus for returning from injury early.|injury_rate=1.0,star_rev=1.0|contender=0.05|-0.05
HLT031|health|Seven-Day IR Minimum|Injured reserve requires at least a week out.|trade_volume=-1.0||0.04
HLT032|health|Three-Day IR Minimum|The injured reserve minimum drops to three days.|trade_volume=1.0,injury_rate=-0.3||0.02
HLT033|health|Goalie Workload Cap|Goalies limited to 65 starts.|injury_rate=-1.0,scoring_pace=0.5||-0.02
HLT034|health|Emergency Goalie Pool|The league funds certified emergency goalies.|opex=0.05,fan=0.4||0.12
HLT035|health|Third Goalie Dressed|Clubs dress a third goalie.|opex=0.2,injury_rate=-0.2||0.0
HLT036|health|Helmet Testing Lab|The league tests every helmet yearly.|injury_rate=-1.0,opex=0.1||0.1
HLT037|health|Sleep Pods on Charters|League-standard sleep pods on team flights.|injury_rate=-0.5,opex=0.3|poor=-0.1|0.0
HLT038|health|Expanded Drug Testing|Expanded performance-drug testing.|fan=0.3,opex=0.1||0.08
HLT039|health|In-Game Load Trackers|Players wear load trackers in games.|injury_rate=-1.0,trade_demand_rate=1.0,opex=0.1||0.02
HLT040|health|Rest After Long Road Trips|One day off after any five-game trip.|injury_rate=-1.0,opex=0.1||0.05
HLT041|health|Physio on Every Charter|A physiotherapist on every road trip.|injury_rate=-0.5,opex=0.2||0.05
HLT042|health|Youth Body-Checking Education|The league funds body-checking education.|fan=0.4,opex=0.1||0.1
HLT043|health|Concussion Insurance Pool|A league pool pays clubs for long concussion absences.|opex=0.3,cap_adjust_m=0.2||0.05
HLT044|health|Four-in-Six Ban|No team plays four games in six nights.|injury_rate=-1.0,rev_all=-0.2||0.05
HLT045|health|Mouthguard Mandate|Mouthguards required for every skater.|injury_rate=-0.5,fan=-0.1||0.08
CPX001|cap|Three-Percent Cap Growth Floor|The cap rises at least 3% a season.|cap_growth=0.6|cap_tight=0.2,poor=-0.2|0.0
CPX002|cap|Four-Percent Cap Growth Ceiling|The cap may rise at most 4% a season.|cap_growth=-0.6|poor=0.2,cap_tight=-0.2|0.0
CPX003|cap|Escrow Overage Rebate|Surplus escrow becomes extra cap next season.|cap_adjust_m=1.5|cap_tight=0.2|0.05
CPX004|cap|Cap Room Carry-Over|Unused cap room up to $1M carries over league-wide.|cap_adjust_m=0.75|cap_room=0.1,rebuilder=0.1|0.04
CPX005|cap|Floor Grace Season|Clubs under the floor get a full season to comply.|floor_ratio=-0.01|poor=0.15,small=0.1|0.04
CPX006|cap|Revenue-Tiered Floor|Low-revenue clubs face a lower floor.|floor_ratio=-0.02,revenue_share=0.3|small=0.2,large=-0.1|0.0
CPX007|cap|Floor at Three-Quarters of the Cap|The lower limit becomes 75% of the upper limit.|floor_ratio=0.03|large=0.1,small=-0.2,poor=-0.2|-0.08
CPX008|cap|Max Salary at 25%|Stars may earn up to 25% of the cap.|max_salary_pct=0.05|contender=0.1,rich=0.1,small=-0.15|-0.12
CPX009|cap|Max Salary at 15%|No player may earn more than 15% of the cap.|max_salary_pct=-0.05|small=0.15,poor=0.1,large=-0.1|-0.1
CPX010|cap|Max Salary at 21%|Max salary nudged up by a point of the cap.|max_salary_pct=0.01|contender=0.05|0.02
CPX011|cap|Max Salary at 19%|Max salary trimmed by a point of the cap.|max_salary_pct=-0.01|small=0.05|0.02
CPX012|cap|Hard Escrow Ceiling|Players never pay more than 10% escrow; owners absorb the rest.|opex=0.3,cap_growth=0.2||0.0
CPX013|cap|Expansion Fees Outside HRR|Expansion fees stay with the owners.|cap_growth=-0.2,value_growth=0.2|rich=0.1|0.05
CPX014|cap|Playoff Gates in HRR|Playoff revenue counts toward the cap.|cap_growth=0.3,playoff_rev=-5.0|cap_tight=0.1,contender=-0.1|0.0
CPX015|cap|Signing Bonuses Charged Up Front|Signing bonuses count against the cap in the year paid.|bonus_pct=-0.04,cap_adjust_m=-0.25|small=0.1,large=-0.1|0.0
CPX016|cap|Cap Recapture Revived|Front-loaded deals carry recapture charges.|bonus_pct=-0.03|small=0.1,rich=-0.1|0.02
CPX017|cap|Amnesty Buyout|Every club gets one cap-free buyout.|cap_adjust_m=1.0,opex=0.3|cap_tight=0.25,poor=-0.1|0.05
CPX018|cap|Buyout Charges Raised|Buyouts carry 75% of the cap hit.|cap_adjust_m=-0.5|cap_room=0.1,cap_tight=-0.15|-0.05
CPX019|cap|Cap Indexed to Inflation|The cap tracks inflation instead of revenue.|cap_growth=-0.2|poor=0.1|0.0
CPX020|cap|Currency Smoothing|Exchange-rate swings smoothed out of the cap.|cap_growth=0.1||0.08
CPX021|cap|Deadline Cap Increase|The cap rises $1M at the trade deadline.|cap_adjust_m=1.0,trade_volume=3.0|cap_tight=0.2|0.02
CPX022|cap|Bonus Overages Charged in Full|All bonus overages hit next season's cap.|cap_adjust_m=-0.5|cap_room=0.05|-0.02
CPX023|cap|Retained Salary Counts to the Floor|Retained salary counts toward the floor; one more slot.|floor_ratio=-0.01,retention_slots=1|poor=0.1|0.05
CPX024|cap|Monthly Escrow|Escrow taken from every paycheque.|opex=-0.2||0.05
CPX025|cap|Upper Limit Rounded Down|The cap rounds down to the nearest $0.5M.|cap_adjust_m=-0.4||0.0
CPX026|cap|Soft Cap Band|Clubs may exceed the cap by $2M and pay a tax into sharing.|cap_adjust_m=1.0,revenue_share=0.6|rich=0.2,poor=-0.2|-0.1
CPX027|cap|Cap Space Sales Banned|Clubs may not sell cap space for picks.|trade_volume=-4.0|cap_room=-0.15|0.0
CPX028|cap|Two-Year Cap Freeze|The cap held flat for two seasons.|cap_growth=-1.0|poor=0.2,cap_tight=-0.3|-0.15
CPX029|cap|New Media Deal Cap Jump|A new national TV deal adds $4M to the cap.|cap_adjust_m=4.0|cap_tight=0.3,poor=-0.15|0.05
CPX030|cap|Small-Market Floor Credit|Small markets may count bonuses toward the floor.|floor_ratio=-0.005,rev_small=0.3|small=0.2|0.0
CPX031|cap|Rookie Cap Discount|Entry-level deals count at 80% against the cap.|cap_adjust_m=0.5|rebuilder=0.15|0.04
CPX032|cap|Veteran Cap Discount|Players 35 and over count 10% less against the cap.|cap_adjust_m=0.3|contender=0.1|0.0
CPX033|cap|Cap Hit Equals Salary|Each year's cap hit equals that year's salary.|bonus_pct=-0.05|small=0.15,large=-0.15|-0.05
CPX034|cap|Five-Year Cap Forecasts|The league publishes five-year cap forecasts.|cap_growth=0.1,fan=0.2||0.12
CPX035|cap|Sharing Tied to Floor Spending|Clubs below the floor lose sharing money.|revenue_share=-0.5,floor_ratio=0.01|large=0.1,poor=-0.15|0.0
CPX036|cap|Attendance-Linked Cap Growth|Cap growth tracks league attendance.|cap_growth=0.2,fan=0.2||0.0
CPX037|cap|Expansion Cap Bump|The cap rises $1.5M the season new clubs join.|cap_adjust_m=1.5,expansion_pressure=5.0|cap_tight=0.15|0.0
CPX038|cap|Arena-Build Floor Relief|Clubs building arenas get a lower floor.|floor_ratio=-0.01,value_growth=0.2|poor=0.1|0.02
CPX039|cap|Bigger Bonus Cushion|The bonus cushion grows to 10% of the cap.|bonus_pct=0.03,cap_adjust_m=0.5|contender=0.1,rebuilder=0.05|0.0
CPX040|cap|Deferred Salary Allowed|Clubs may defer up to 10% of salaries.|cap_adjust_m=0.75,opex=0.3|rich=0.15,poor=-0.15|-0.05
CNX001|contracts|Nine-Year Re-Sign Term|Clubs may re-sign their own players for nine years.|max_term_own=9|large=0.1,contender=0.05,small=-0.1|-0.08
CNX002|contracts|Five-Year Re-Sign Term|Re-signing terms capped at five years.|max_term_own=5|small=0.1,poor=0.1,large=-0.1|-0.08
CNX003|contracts|Eight-Year Free-Agent Term|Free agents may sign eight-year deals.|max_term_ufa=8|large=0.15,rich=0.1,small=-0.15|-0.1
CNX004|contracts|Four-Year Free-Agent Term|Free-agent deals capped at four years.|max_term_ufa=4|small=0.15,poor=0.1,large=-0.15|-0.08
CNX005|contracts|Seven Years for Everyone|Every contract limited to seven years.|max_term_own=7,max_term_ufa=7|medium=0.05|0.0
CNX006|contracts|Ten-Year Franchise Deal|Clubs may re-sign their own players for up to ten years.|max_term_own=10|large=0.1,contender=0.1,small=-0.1|-0.12
CNX007|contracts|League Minimum Up $250K|The league minimum rises by $250K.|min_salary_m=0.25|all=-0.1,cap_room=0.05|-0.05
CNX008|contracts|Minimum Indexed to the Cap|The minimum salary grows with the cap: +$75K now.|min_salary_m=0.075||0.05
CNX009|contracts|Rookie Minimum Lowered|The first-year minimum drops $50K.|min_salary_m=-0.05|rebuilder=0.1|0.0
CNX010|contracts|Signing Bonus Room Doubled|Clubs above the revenue floor can pay twice the bonuses.|bonus_pct=0.08|large=0.2,rich=0.15,small=-0.2|-0.08
CNX011|contracts|Signing Bonus Ban|No signing bonuses at all.|bonus_pct=-0.12,fa_bonus_demand=-8.0|small=0.25,poor=0.2,large=-0.25|-0.1
CNX012|contracts|Higher Bonus Revenue Bar|Clubs need $10M more revenue to pay bonuses.|bonus_floor_m=10.0|large=0.15,small=-0.2|-0.03
CNX013|contracts|Lower Bonus Revenue Bar|Clubs need $20M less revenue to pay bonuses.|bonus_floor_m=-20.0|small=0.25,medium=0.1,large=-0.1|0.0
CNX014|contracts|Bonus Demand Cooling Period|Agents may only ask for bonuses after day three of free agency.|fa_bonus_demand=-6.0|small=0.15|0.04
CNX015|contracts|Star Bonus Expectations|More free agents expect bonuses (+8%).|fa_bonus_demand=8.0|large=0.15,small=-0.15|-0.05
CNX016|contracts|Offer Sheet Compensation Halved|Offer-sheet pick compensation cut in half.|trade_demand_rate=3.0,trade_volume=2.0|cap_room=0.15,cap_tight=-0.15|0.0
CNX017|contracts|Offer Sheet Compensation Raised|Offer sheets cost extra picks.|trade_demand_rate=-2.0|cap_tight=0.1,cap_room=-0.1|0.0
CNX018|contracts|Free Agency at 25|Players reach unrestricted free agency at 25.|fa_bonus_demand=4.0,trade_demand_rate=5.0|large=0.1,small=-0.15|-0.08
CNX019|contracts|Free Agency at 28|Players must be 28 to reach unrestricted free agency.|fa_bonus_demand=-4.0,trade_demand_rate=-3.0|small=0.15,large=-0.1|-0.05
CNX020|contracts|One Arbitration Per Career|Players can file for arbitration once in their career.|trade_demand_rate=-2.0,min_salary_m=-0.01|all=0.05|0.0
CNX021|contracts|Club-Elected Arbitration Expanded|Clubs may take any RFA to arbitration.|trade_demand_rate=3.0|cap_tight=0.1|0.0
CNX022|contracts|No-Trade Clauses at 25|NTCs allowed from age 25.|trade_demand_rate=5.0,trade_volume=-3.0|all=-0.05|-0.03
CNX023|contracts|No-Trade Clauses at 29|NTCs only after age 29.|trade_demand_rate=-3.0,trade_volume=3.0|all=0.05|0.02
CNX024|contracts|Front-Load Limit|A deal's first year can't exceed its lowest year by more than 35%.|bonus_pct=-0.03|small=0.1|0.03
CNX025|contracts|Escalating Contracts Allowed|Salaries may rise every year without limit.|bonus_pct=0.02,cap_growth=-0.1|rebuilder=0.05|0.0
CNX026|contracts|Two-Way Minimum Raised|AHL salaries on two-way deals rise.|opex=0.2,min_salary_m=0.02||0.03
CNX027|contracts|One-Way Deals at 26|Players 26 and over sign one-way deals only.|opex=0.2|poor=-0.1|0.0
CNX028|contracts|Depth Contracts Guaranteed|All contracts fully guaranteed through buyouts.|opex=0.3,cap_adjust_m=-0.25||0.0
CNX029|contracts|Player Option Years|Players may hold an option on the final year.|fa_bonus_demand=2.0,trade_demand_rate=-2.0|large=0.05|0.0
CNX030|contracts|Club Option Years|Clubs may hold an option on the final year.|trade_demand_rate=2.0|small=0.1,poor=0.05|0.0
CNX031|contracts|Performance Pay Pool|A year-end bonus pool for top performers.|star_rev=2.0,opex=0.2||0.05
CNX032|contracts|Loyalty Bonus Program|Players who re-sign with their club get a league bonus.|trade_demand_rate=-5.0,opex=0.2||0.08
CNX033|contracts|Extensions at the Deadline|Extensions may be signed at the trade deadline.|trade_demand_rate=-3.0,trade_volume=-2.0||0.05
CNX034|contracts|Final-Year Extensions Only|Extensions only in the final season of a deal.|trade_demand_rate=3.0||0.0
CNX035|contracts|Two-Year Entry-Level Deals|Entry-level deals run two years.|fa_bonus_demand=1.0,trade_demand_rate=2.0|rebuilder=-0.1,large=0.05|-0.05
CNX036|contracts|Four-Year Entry-Level Deals|Entry-level deals run four years.|trade_demand_rate=-2.0|rebuilder=0.15,small=0.05|0.0
CNX037|contracts|No ELC Performance Bonuses|No performance bonuses on entry-level deals.|cap_adjust_m=0.25,star_rev=-1.0|rebuilder=0.1|0.0
CNX038|contracts|35-Plus Rule Repealed|Retiring older players clear their cap hit.|cap_adjust_m=0.3|contender=0.1|0.05
CNX039|contracts|Contract Database Published|Every contract term published.|fan=0.3,fa_bonus_demand=-2.0||0.1
CNX040|contracts|Agent Certification Exam|Agents must pass a league exam.|fa_bonus_demand=-2.0,trade_demand_rate=-1.0||0.08
CNX041|contracts|Holdout Fines Doubled|Players who hold out lose double pay.|trade_demand_rate=-4.0,fan=-0.1||0.05
CNX042|contracts|Deferred Pay Interest|Players earn interest on deferred pay.|opex=0.2||0.0
CNX043|contracts|Veteran Retirement Exit|Players 33 and over can retire with no cap penalty.|cap_adjust_m=0.25,trade_demand_rate=-1.0|contender=0.05|0.05
CNX044|contracts|Bonus Payment Date Moved|Bonuses paid on July 15th instead of July 1st.|opex=-0.1||0.08
CNX045|contracts|Two-Year Minimum Term|Free agents must sign for at least two years.|fa_bonus_demand=-1.0,trade_volume=-1.0||0.0
"""


def _parse_kv(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        k, _sep, v = part.partition("=")
        out[k.strip()] = float(v)
    return out


def _parse(data: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in data.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) != 7:
            raise ValueError(f"bad rule row: {line[:60]}")
        rid, cat, title, summary, effects, lean, base = parts
        rows.append({
            "id": rid.strip(), "cat": cat.strip(), "title": title.strip(), "summary": summary.strip(),
            "effects": _parse_kv(effects), "lean": _parse_kv(lean), "base": float(base or 0.0),
        })
    return rows


def extra_rules(make: Callable[..., Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build the 500 extra rules with the base catalog's constructor."""
    out: List[Dict[str, Any]] = []
    for row in _parse(DATA_A) + _parse(DATA_B):
        out.append(make(row["id"], row["cat"], row["title"], row["summary"], row["effects"], row["lean"], row["base"]))
    return out
