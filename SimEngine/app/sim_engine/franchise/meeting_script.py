"""Written dialogue for player meetings (data only).

Three layers, all in the player's own first-person voice:

REPLIES  "itype:answer_id" -> (warm lines, cold lines)
         His answer to what the GM actually said. Warm = it landed, cold = it didn't.
ASKS     itype -> {"ask": lines, "opts": [...]}
         After a warm or guarded reply he asks for one specific thing. The GM's second
         (last) choice answers that ask: yes / partly / no. Each option carries the GM's
         spoken line and his goodbye, which names what was agreed.
HOT_CLOSE  closing lines for the tempers-flared branch, keyed by the GM's last move.

Placeholders filled from the save when known (a line is skipped if one is missing):
{first} {toi} {line} {pts} {gp} {g} {task} {role}
"""

from __future__ import annotations

from typing import Dict, List, Tuple

Lines = Tuple[str, ...]

# ---------------------------------------------------------------------------
# Replies to the GM's first answer
# ---------------------------------------------------------------------------

REPLIES: Dict[str, Tuple[Lines, Lines]] = {
    # --- role / ice time ---
    "discuss_ice_time:honest": (
        ("Okay. That's fair. I'd rather know what to fix than guess.", "Alright. Puck battles and the details. I can do that."),
        ("So it's on me. I figured you'd say that.", "That's the same thing the coaches said. I'm doing those things."),
    ),
    "discuss_ice_time:promise": (
        ("A few weeks. Okay. I'll be ready when it comes.", "That's all I wanted to hear. I'll make it easy for you."),
        ("A few weeks is a long time when you're sitting.", "I've heard 'give me a few weeks' before."),
    ),
    "discuss_ice_time:firm": (
        ("Fair. Then I'll make it hard on you.", "Okay. No complaints from me. I'll just play."),
        ("So nothing changes. Got it.", "Make it hard on you? I've been trying to."),
    ),
    "offer_increased_ice_time:commit": (
        ("Seriously? Okay. I won't waste them.", "Thank you. I've been waiting for this."),
        ("Okay. We'll see if the sheet says the same thing.", "I'll believe it when I see the minutes."),
    ),
    "offer_increased_ice_time:conditional": (
        ("Fair enough. I'll keep it up.", "That's motivation. I'll take it."),
        ("So I have to keep proving it while someone else gets the ice.", "I thought I already earned it. That's why I'm here."),
    ),
    "explain_reduced_ice_time:transparent": (
        ("Okay. I don't love it, but I appreciate you telling me straight.", "That makes sense. I'd rather know than wonder."),
        ("A 'different look' doesn't tell me what I did wrong.", "So it's not even about me. That's worse, honestly."),
    ),
    "explain_reduced_ice_time:performance": (
        ("Yeah. I know I've been off. I'll get it back.", "That's fair. I haven't been good enough lately."),
        ("One stretch and the minutes go? Other guys get longer leashes.", "I don't think my game's dropped that much."),
    ),
    "explain_reduced_ice_time:deflect": (
        ("Alright. I'll go talk to the coach then.", "Okay. I'll ask him."),
        ("You're the GM. You know why.", "So nobody's going to tell me. Great."),
    ),
    "discuss_line_assignment:top_six": (
        ("That's what I needed to hear. Tell me what it takes.", "Top six. Okay. I'm in."),
        ("A 'path' isn't a spot. When do I actually get there?", "I've heard about this path for a while."),
    ),
    "discuss_line_assignment:depth": (
        ("Okay. I'd rather know. I'll be the best depth guy you've got.", "Fair. Someone has to do the dirty work. I'll own it."),
        ("Depth. Okay. That's not what I came here to be.", "So that's my ceiling here. Good to know."),
    ),
    "discuss_promotion:support": (
        ("Yes. Finally. I won't let it slip.", "Thank you. I've been pushing for this."),
        ("I'll believe it when I see the line chart.", "Okay. When, though?"),
    ),
    "discuss_promotion:wait": (
        ("Okay. Then I'll keep pushing until you have no choice.", "Fair. I'll force your hand."),
        ("Not yet? What else do I have to show you?", "I've been doing exactly that for a month."),
    ),
    "discuss_demotion:accountability": (
        ("Fair. I've been sloppy. I'll fix it.", "Yeah. I know what the details are. I'll be back up there."),
        ("The details? I'm not the only one making mistakes on that line.", "So it's all on me."),
    ),
    "discuss_demotion:support": (
        ("Okay. I can handle a short-term thing.", "Thanks. Good to know it's not permanent."),
        ("Everyone says short-term.", "Hang in there. Sure."),
    ),
    "offer_pp_opportunity:promise_pp": (
        ("Finally. I'll make it count.", "That's huge. Thank you. I've got ideas for that unit."),
        ("Okay. Let's see if the coaches actually put me out there.", "A 'real look' better mean more than one shift."),
    ),
    "offer_pp_opportunity:deny": (
        ("Fair. I'll show you at five-on-five first.", "Okay. I'll earn it the right way."),
        ("I thought you called me in to give me the power play.", "So why bring it up at all?"),
    ),
    "explain_pp_removal:explain": (
        ("Okay. The unit wasn't working. I get it.", "Fair. I wasn't helping it either."),
        ("And I'm the one who pays for it?", "We weren't producing as five guys. Only one of us lost a spot."),
    ),
    "explain_pp_removal:challenge": (
        ("Okay. I'll win it back.", "Watch me. I'll be back on it in a week."),
        ("Win it back how? I'm not out there to show you.", "Easy to say when I'm on the bench for it."),
    ),
    "offer_pk_opportunity:yes": (
        ("The kill? Yeah. I'll block anything.", "Love it. That's a job I can own."),
        ("PK minutes. Okay. Is that instead of something else?", "I'll do it. I hope it's not a consolation prize."),
    ),
    "offer_pk_opportunity:no": (
        ("Fair. I'll bring the energy at five-on-five.", "Okay. Even strength it is."),
        ("Then why bring up the kill?", "So I don't get the PK either. Okay."),
    ),
    "discuss_healthy_scratch:development": (
        ("Okay. A reset. I can treat it that way.", "Thanks. I needed to hear it's not a punishment."),
        ("It sure feels like a punishment from the press box.", "A reset. In a suit. Okay."),
    ),
    "discuss_healthy_scratch:performance": (
        ("Yeah. I know. I've been bad. I'll fix it.", "Fair. I'd rather hear it straight."),
        ("I know what needs to improve? Tell me, because I'm not sure.", "Other guys have played worse and dressed."),
    ),
    "discuss_return_lineup:welcome": (
        ("Good. I'm ready. Let's go.", "That's all I wanted. Thanks for holding it for me."),
        ("Good. I was worried someone took it.", "Okay. Let's see if the coaches agree."),
    ),
    "discuss_return_lineup:earn": (
        ("Fair. The guys played well. I'll take it back.", "Okay. I'll earn it. Wouldn't want it handed to me."),
        ("I got hurt playing for this team. Now I have to start over?", "That's a tough way to welcome a guy back."),
    ),
    "discuss_goalie_workload:promise_starts": (
        ("A run. Okay. I'll take the net and keep it.", "That's what I wanted. Thank you."),
        ("How long's a run? Three games?", "Okay. As long as one bad night doesn't end it."),
    ),
    "discuss_goalie_workload:platoon": (
        ("Okay. At least I know how it works.", "Fair. I'll make the big games mine."),
        ("Splitting it means neither of us gets a rhythm.", "So I'm half a starter."),
    ),
    # --- performance ---
    "praise_performance:specific": (
        ("That means a lot. Nobody notices the stuff away from the puck.", "Thanks. Coming from you, that's huge."),
        ("Thanks. I'd rather it showed up in my ice time, though.", "Appreciate it. Points are what get noticed, though."),
    ),
    "praise_performance:team": (
        ("Thanks. That's how I want to play.", "Means a lot. The guys make it easy."),
        ("Appreciate it.", "Okay. Thanks."),
    ),
    "challenge_performance:direct": (
        ("Yeah. I know. I needed that.", "Fair. I've been coasting. It stops now."),
        ("I'm not coasting out there.", "That's easy to say from up here."),
    ),
    "challenge_performance:support": (
        ("Okay. I'll take all of it. Let's fix this.", "Thanks. I want to be better too."),
        ("Whatever I need? I need minutes with guys who can finish.", "Okay. We'll see."),
    ),
    "ask_struggles:listen": (
        ("Honestly? I'm in my own head. Every shot feels like it has to go in.", "It's been a lot. Thanks for asking instead of telling me."),
        ("I don't really know. That's the problem.", "It's just hockey. I'll figure it out."),
    ),
    "ask_struggles:fix_role": (
        ("A new role might help. I'm open to it.", "Okay. Something different could free me up."),
        ("So you're moving me again.", "Changing my role isn't going to fix my confidence."),
    ),
    "discuss_scoring_drought:stay_confident": (
        ("Thanks. I needed someone to say that.", "Okay. I'll keep shooting."),
        ("Keep shooting. I've been shooting.", "That's what everyone says."),
    ),
    "discuss_scoring_drought:usage_change": (
        ("I'll try anything at this point.", "A new look might be good. I'm in."),
        ("So I lose my spot because the puck won't go in.", "Moving me around isn't going to help."),
    ),
    "discuss_defensive_struggles:film": (
        ("Okay. I'll be there. I know I've been chasing.", "Good. Show me what you're seeing."),
        ("I watch film already.", "More video. Okay."),
    ),
    "discuss_defensive_struggles:pairing": (
        ("That would help a lot, honestly.", "Okay. A steadier partner would let me play my game."),
        ("So it's my partner's fault now?", "I don't need someone to cover for me."),
    ),
    "discuss_improvement:encourage": (
        ("Thanks. It feels good out there right now.", "Will do. I'm not changing a thing."),
        ("Okay.", "Sure. Thanks."),
    ),
    "discuss_improvement:raise_bar": (
        ("I like that. I want great too.", "Okay. Let's see how far I can push it."),
        ("Can I enjoy it for one week first?", "There's always more, huh."),
    ),
    "discuss_consistency:routine": (
        ("That's fair. My routine's been all over the place.", "Okay. Same prep, same habits. I can do that."),
        ("My routine isn't the problem.", "I prepare the same every night."),
    ),
    "discuss_consistency:pressure": (
        ("Honestly, that helps. It has felt like a lot.", "Thanks. I didn't know that was showing."),
        ("I'm not cracking under pressure.", "It's not about pressure."),
    ),
    "discuss_playoff_performance:belief": (
        ("Thanks. I've been waiting for this all year.", "That means a lot right now."),
        ("Belief doesn't give me minutes on the power play.", "Thanks. I'll be ready."),
    ),
    "discuss_playoff_performance:prep": (
        ("Good. I want nothing left to chance either.", "Agreed. Let's lock it in."),
        ("I know how to prepare for the playoffs.", "My prep's fine."),
    ),
    # --- development ---
    "explain_ovr_rising:credit": (
        ("Thanks. I worked hard this summer. Feels good to hear it.", "That means a lot. I'll keep going."),
        ("Appreciate it.", "Thanks. Now I'd like the role to match."),
    ),
    "explain_ovr_rising:demand_more": (
        ("I like that. I'm not satisfied either.", "Good. I want to find out too."),
        ("Can I at least enjoy it for a second?", "Okay. Always more."),
    ),
    "explain_ovr_falling:support": (
        ("Thanks. I needed to hear that.", "Okay. I'll get through it."),
        ("Slumps happen. Sure.", "I'll figure it out."),
    ),
    "explain_ovr_falling:accountability": (
        ("Fair. My prep hasn't been where it should be.", "Yeah. I'll be ready every day."),
        ("I show up ready.", "That's not what this is."),
    ),
    "discuss_development_plan:clear_plan": (
        ("That's exactly what I wanted. Something real.", "Okay. Now I know what I'm chasing."),
        ("Okay. Sounds like a lot of meetings.", "We'll see if it's different from last year's plan."),
    ),
    "discuss_development_plan:skills": (
        ("Good. Extra work is fine with me.", "Okay. When do I start?"),
        ("Skills coaches. Okay.", "I was hoping for more than extra drills."),
    ),
    "set_development_priority:skating": (
        ("Skating. Okay. I'll be on the ice before everyone.", "Makes sense. A step quicker changes everything."),
        ("Skating? I thought my skating was fine.", "Okay. If you say so."),
    ),
    "set_development_priority:hockey_sense": (
        ("My reads. Yeah. I can feel when I'm a step late.", "Okay. I'll watch more tape."),
        ("Hockey sense? That's a tough one to just work on.", "I'll try."),
    ),
    "set_development_priority:shooting": (
        ("My release. Yeah. I'll take a few hundred a day.", "Shooting. I like that. I want to score more."),
        ("I thought my shot was a strength.", "Okay. I'll shoot pucks."),
    ),
    "set_development_priority:defence": (
        ("Own zone. Fair. That's what keeps me off the ice late.", "Okay. I'll own the defensive side."),
        ("Defence. Not exactly what I was hoping for.", "So you don't trust me in my own end."),
    ),
    "set_development_priority:strength": (
        ("The weight room. Yeah. I get pushed off pucks.", "Okay. I'll live in the gym."),
        ("I'm strong enough.", "Okay. More lifting."),
    ),
    "set_development_priority:faceoffs": (
        ("Draws. Yeah. That's a quick way to earn trust.", "Okay. I'll take a hundred a day."),
        ("Faceoffs? That's it?", "Okay. Draws."),
    ),
    "set_development_priority:rebounds": (
        ("Rebounds. Yeah. Too many second chances lately.", "Okay. I'll work on swallowing pucks."),
        ("My rebound control's fine.", "I'll look at the tape."),
    ),
    "set_development_priority:positioning": (
        ("Positioning. Yeah. I've been scrambling.", "Okay. I'll stay square and let it hit me."),
        ("My positioning's not the issue.", "I'll look at it."),
    ),
    "discuss_stalled_development:patience": (
        ("Okay. I'll try not to get in my own head about it.", "That helps. I've been impatient."),
        ("Be patient. Sure.", "I've been patient. That's the problem."),
    ),
    "discuss_stalled_development:opportunity": (
        ("That's all I need. A real shot.", "Okay. I'll be ready."),
        ("When?", "I'll believe it when it comes."),
    ),
    "discuss_aging_decline:veteran_role": (
        ("I'd like that. The kids remind me why I love this.", "Okay. I can be that guy."),
        ("So I'm a mentor now. Not a player.", "That sounds like a nice way to say 'you're done'."),
    ),
    "discuss_aging_decline:honest": (
        ("I appreciate that. I'd rather hear it from you.", "Fair. I'm not blind."),
        ("Might come down. Or will?", "That stings, even if I saw it coming."),
    ),
    "discuss_conditioning:staff": (
        ("Good. My legs need it.", "Okay. I'll do whatever they ask."),
        ("I don't need babysitting.", "Fine. I'll see them."),
    ),
    "discuss_conditioning:expect": (
        ("Fair. I'll be ready.", "Understood. Every night."),
        ("I am ready every night.", "So you think I'm out of shape."),
    ),
    "discuss_injury_recovery:no_rush": (
        ("Thanks. Honestly, that's a weight off.", "That means a lot. I'll do it right."),
        ("I don't want to rush, but I hate watching.", "I'll follow the plan."),
    ),
    "discuss_injury_recovery:timeline": (
        ("Good. A target helps.", "Okay. Knowing the plan helps my head."),
        ("I'll be back before that.", "That's longer than I hoped."),
    ),
    # --- contract ---
    "ask_extension_interest:interested": (
        ("Good. I want to be here. Let's get it done.", "That's what I hoped. I'll tell my agent."),
        ("Okay. Let's see what your number looks like.", "Interested isn't an offer."),
    ),
    "ask_extension_interest:noncommittal": (
        ("Fair. I'll keep my options open too.", "Okay. I get it, it's business."),
        ("So you're not sure about me.", "Then I guess my agent keeps listening to other teams."),
    ),
    "long_term_plans:commit": (
        ("That means a lot. I want to be here.", "Good. I want to win here."),
        ("Words are nice. A contract's better.", "I'll wait for the offer."),
    ),
    "long_term_plans:cautious": (
        ("Fair. I'll keep it up.", "Okay. I'll keep earning it."),
        ("So it's conditional.", "That's not exactly a commitment."),
    ),
    "future_uncertain:honest": (
        ("I appreciate you being straight about it.", "Thanks for not dressing it up."),
        ("The cap. Always the cap.", "That's hard to hear."),
    ),
    "future_uncertain:soften": (
        ("Thanks. That means something.", "I appreciate that."),
        ("Respect won't pay my mortgage.", "I'll think about it."),
    ),
    "discuss_role_next_season:top_role": (
        ("A top role. I'll be ready for it.", "That's what I wanted to hear."),
        ("Plan for it? Put it in writing.", "We'll see what happens in the summer."),
    ),
    "discuss_role_next_season:depth": (
        ("Okay. I'd rather know now.", "Fair. At least I can plan."),
        ("Depth. Okay. My agent will want to hear that.", "Then I should think about my options."),
    ),
    "discuss_upcoming_fa:priority": (
        ("Good. I want to stay. Let's not let it get to July.", "That's what I hoped. Let's talk numbers."),
        ("A priority. Okay. Then make an offer.", "We'll see if the offer says that."),
    ),
    "discuss_upcoming_fa:open": (
        ("Fair. Let's keep talking.", "Okay. No rush."),
        ("Keep talking means no offer.", "Then I'll keep my options open too."),
    ),
    "hometown_discount:ask": (
        ("I want to stay. I'll think about it.", "Okay. If it keeps a winning team together, maybe."),
        ("So you want me to pay for the cap mess.", "I knew that was coming."),
    ),
    "hometown_discount:respect": (
        ("I appreciate that. Let's find something fair.", "Good. That makes this easier."),
        ("Okay. Then let's see the number.", "Glad we agree. My agent will be in touch."),
    ),
    "contract_expectations:listen": (
        ("Okay. Honestly? I want term more than money.", "Fair. I'll be honest with you too."),
        ("My agent will give you the number.", "I'd rather hear your offer first."),
    ),
    "contract_expectations:anchor": (
        ("Okay. At least you're honest. Let's see how far apart we are.", "Fair. Let's keep talking."),
        ("Lower than I'd like? We'll see what other teams think.", "That's not a great start."),
    ),
    "discuss_retirement:support": (
        ("Thanks. That makes it easier to think about.", "That means a lot. Really."),
        ("Okay.", "I haven't decided anything."),
    ),
    "discuss_retirement:win_now": (
        ("One more run. Yeah. I'd like that.", "If we're going for it, I'm in."),
        ("I've heard that before.", "Only if the team's actually good enough."),
    ),
    # --- trade / AHL ---
    "ask_trade_welcome:explore": (
        ("Okay. Quietly. I'd appreciate that.", "Fair. Maybe a fresh start is right."),
        ("So you want me gone.", "Wow. Okay."),
    ),
    "ask_trade_welcome:stay": (
        ("Good. I'd like to stay too.", "Okay. Thanks for saying it."),
        ("Then why ask?", "I'm confused, honestly."),
    ),
    "tell_being_shopped:honest": (
        ("I appreciate you telling me first.", "Okay. At least I didn't read it online."),
        ("So I'm on the block.", "Great. My wife's going to love this."),
    ),
    "tell_being_shopped:context": (
        ("Okay. I know how this works.", "Fair. It's business."),
        ("Business. Right. It's my family that moves.", "That's easy to say when you're not packing."),
    ),
    "reassure_not_shopped:firm": (
        ("Good. That's all I needed. I can stop checking my phone.", "Thank you. My family's been asking."),
        ("Okay. I'll believe it when the deadline passes.", "Everyone says that."),
    ),
    "reassure_not_shopped:vague": (
        ("Okay. Thanks.", "Alright."),
        ("Nothing imminent isn't 'no'.", "That's not really reassuring."),
    ),
    "ask_ntc_waiver:destination": (
        ("Okay. I'll listen. Where are we talking about?", "I'm not saying yes, but I'll hear it."),
        ("I negotiated that clause so I'd never have this conversation.", "No. Not without a very good reason."),
    ),
    "ask_ntc_waiver:withdraw": (
        ("Thank you. I appreciate that.", "Good. Let's forget it happened."),
        ("Okay. But I won't forget you asked.", "Fine."),
    ),
    "ask_preferred_destinations:listen": (
        ("Okay. A contender, close to home if possible.", "Somewhere I can win. I'll give you a short list."),
        ("I'll have my agent send it.", "I'd rather not be in this situation at all."),
    ),
    "ask_preferred_destinations:discourage": (
        ("Okay. Fine by me.", "Good. I'd rather play here anyway."),
        ("You brought it up.", "Okay, I'll drop it."),
    ),
    "discuss_ahl_assignment:development": (
        ("Okay. Big minutes. I'll go get better.", "Fair. I'll be back soon."),
        ("It still feels like a demotion.", "How long?"),
    ),
    "discuss_ahl_assignment:performance": (
        ("Fair. I haven't been good enough. I'll earn it back.", "Okay. I'll make you call me up."),
        ("I haven't gotten enough minutes to show anything.", "So it's on me. Okay."),
    ),
    "discuss_call_up:soon": (
        ("Soon. Okay. I'll keep it up.", "That's huge. I'll be ready."),
        ("Soon could mean anything.", "Okay. I'll believe it when I'm on the plane."),
    ),
    "discuss_call_up:work": (
        ("Fair. I'll keep pushing.", "Okay. I'll make it impossible to keep me down here."),
        ("What else do I have to do down here?", "Not yet. Again."),
    ),
    "address_trade_rumors:deny": (
        ("Good. That's all I needed to hear.", "Okay. I'll tune it out."),
        ("Okay. I'll believe it after the deadline.", "That's what they always say."),
    ),
    "address_trade_rumors:acknowledge": (
        ("Okay. I appreciate the honesty.", "Fair. Just don't let me find out online."),
        ("Anything can happen. Great.", "That's not reassuring."),
    ),
    # --- room / team ---
    "ask_coaching_relationship:mediate": (
        ("That would help. We just need to clear the air.", "Okay. I'm open to that."),
        ("That's going to be awkward.", "I don't think a meeting fixes it."),
    ),
    "ask_coaching_relationship:support_coach": (
        ("Fair. I'll figure it out with him.", "Okay. I'll make it work."),
        ("So I'm on my own.", "Okay. Then I'll keep my head down."),
    ),
    "ask_locker_room_concerns:listen": (
        ("Okay. Some of the young guys feel like the vets get a pass.", "Alright. Honestly, guys don't know where they stand."),
        ("It's not really my place.", "Nothing I want to put on anyone."),
    ),
    "ask_locker_room_concerns:address": (
        ("Good. It needs addressing.", "Okay. The leaders should hear it."),
        ("Just don't say it came from me.", "That could make it worse if it's handled wrong."),
    ),
    "discuss_leadership:praise_leader": (
        ("Thanks. That means a lot.", "I appreciate that. I take it seriously."),
        ("Appreciate it.", "Okay. Thanks."),
    ),
    "discuss_leadership:expect_more": (
        ("Fair. I can be louder.", "Okay. I'll step up."),
        ("I lead my way.", "I'm not a yeller."),
    ),
    "ask_mentor_younger:assign": (
        ("I'd be glad to. Who is it?", "Happy to. Somebody did it for me."),
        ("Okay. I've got my own game to worry about, though.", "Fine."),
    ),
    "ask_mentor_younger:decline": (
        ("No problem. Offer stands if you change your mind.", "Fine by me."),
        ("Then why bring it up?", "Okay, forget it."),
    ),
    "personal_wellbeing_check:support": (
        ("Thanks. I might take you up on that.", "Honestly, that helps. It's been a rough few weeks."),
        ("I'm fine.", "I'll handle it."),
    ),
    "personal_wellbeing_check:boundaries": (
        ("Thanks. I appreciate that.", "That means a lot. Really."),
        ("Okay.", "Thanks."),
    ),
    "repair_relationship:apologize": (
        ("I appreciate that. Not a lot of GMs would say it.", "Okay. Thanks. Let's move forward."),
        ("Words are easy.", "Okay. We'll see."),
    ),
    "repair_relationship:reset": (
        ("Okay. Clean slate. I like that.", "Fair. Let's start over."),
        ("Reset. Sure.", "It's not that simple."),
    ),
    "repair_relationship:deflect": (
        ("Okay. Maybe I am.", "Alright."),
        ("Bigger than it is? You don't see it from my side.", "That's exactly the problem."),
    ),
    "address_social_post:take_it_down": (
        ("Yeah. That was dumb. I'll take it down.", "Fair. I didn't think it through."),
        ("An apology? For that?", "Okay. It's coming down."),
    ),
    "address_social_post:back_him": (
        ("Thanks. I didn't expect that.", "That means a lot."),
        ("Okay.", "Good."),
    ),
    "address_social_post:posting_ban": (
        ("Fair. Probably good for me anyway.", "Okay. Phone's in the bag."),
        ("Three weeks? For one post?", "That's a bit much."),
    ),
    "address_social_post:laugh_off": (
        ("Ha. Yeah. Okay.", "Thanks. I'll be smarter next time."),
        ("Okay.", "Sure."),
    ),
    # --- meetings he asked for ---
    "request_more_ice:honest": (
        ("Okay. That's fair. Now I know what to work on.", "Thanks. I'd rather hear it straight."),
        ("So I'm not there yet.", "I thought I was doing those things."),
    ),
    "request_more_ice:promise": (
        ("That's all I wanted. I won't waste it.", "Thanks. You'll see."),
        ("Okay. Five games. I'm counting.", "I'll believe it when I see the sheet."),
    ),
    "request_more_ice:decline": (
        ("Fair. I'll keep earning it.", "Okay. I'll make it obvious."),
        ("I came in here because I thought I had earned it.", "So nothing changes."),
    ),
    "request_pp_time:honest": (
        ("Okay. At least I know where I sit.", "Fair. I'll be ready if a spot opens."),
        ("So I'm behind everyone.", "That's not what I hoped."),
    ),
    "request_pp_time:promise": (
        ("Thank you. I'll make it count.", "That's huge. I won't let the unit down."),
        ("Eight games is a long time.", "Okay. I'm counting."),
    ),
    "request_pp_time:decline": (
        ("Okay. Fair.", "I'll keep showing you at five-on-five."),
        ("The power play's dead and you won't try something new?", "Okay. Got it."),
    ),
    "request_starting_role:honest": (
        ("Okay. At least I know how you're thinking.", "Fair. I'll make it hard for you."),
        ("So he's still the guy.", "I'll keep pushing."),
    ),
    "request_starting_role:promise": (
        ("Thank you. I'll take the net and keep it.", "That's all I wanted."),
        ("A run of starts. How many?", "Okay. As long as one bad game doesn't end it."),
    ),
    "request_starting_role:decline": (
        ("Okay. I'll be ready when you need me.", "Fair. I'll keep pushing."),
        ("I'm playing better than him.", "So I sit."),
    ),
    "contract_clarity:honest": (
        ("Okay. That's what I needed. Thanks.", "Fair. Now I can stop wondering."),
        ("That's not the answer I wanted.", "Okay. At least I know."),
    ),
    "contract_clarity:commit": (
        ("Good. That takes a weight off.", "Thanks. I want to be here."),
        ("Okay. When does my agent get a call?", "Words, though."),
    ),
    "contract_clarity:deflect": (
        ("Okay. I'll wait.", "Fair. It's early."),
        ("So I keep wondering.", "That's what my agent said you'd say."),
    ),
    "winning_concern_meeting:honest": (
        ("Okay. That's more of a plan than I expected.", "Fair. I can get behind that."),
        ("That sounds like a long rebuild.", "I don't have five years."),
    ),
    "winning_concern_meeting:promise_compete": (
        ("That's what I wanted to hear. Let's go.", "Good. I'm in."),
        ("I'll believe it at the deadline.", "Okay. We'll see."),
    ),
    "winning_concern_meeting:dismiss": (
        ("Fair. I'll focus on my game.", "Got it."),
        ("I care about winning and that's the answer?", "Got it."),
    ),
    "development_meeting:honest": (
        ("That's exactly what I wanted. Thanks.", "Okay. Now I know what to chase."),
        ("Okay. Sounds like a lot.", "We'll see."),
    ),
    "development_meeting:skills_focus": (
        ("Good. When do I start?", "Okay. Extra work is fine."),
        ("Okay. Drills.", "I was hoping for more."),
    ),
    "development_meeting:wait": (
        ("Okay. I'll stay patient.", "Fair."),
        ("Patient. Sure.", "That's not a plan."),
    ),
    # --- bonus answers ---
    "ctx_what_he_means": (
        ("That means a lot. Thanks.", "Okay. I needed to hear that."),
        ("Nice words. Doesn't change the problem.", "I know what I am to the team. I'm asking about the problem."),
    ),
    "ctx_earn_it": (
        ("Fair. I'd rather hear it straight than guess.", "Okay. I'll earn it."),
        ("So I'm one bad game from the press box. Got it.", "Message received."),
    ),
    "ctx_development_path": (
        ("That's exactly what I wanted. A real plan.", "Okay. I know what to work on now."),
        ("Okay.", "Sounds good on paper."),
    ),
    "ctx_contract_results": (
        ("Fair. I know I haven't earned it lately.", "Yeah. I feel it too."),
        ("You're throwing my contract at me?", "I didn't set the number. You did."),
    ),
    "ctx_lead_it": (
        ("Okay. I can do that. The guys will listen.", "Fair. I'll handle it."),
        ("So now it's my job to fix it.", "That's not really my place."),
    ),
}

# Habit changes and role pacts share three answers; their replies fill in what's asked.
STYLE_REPLIES: Dict[str, Tuple[Lines, Lines]] = {
    "habit:explain": (
        ("Okay, that video's pretty convincing. I'll try {task}.", "Fair. I didn't see it like that. I'll work on it."),
        ("Numbers don't show everything.", "I see what you're showing me. I'm not sure it's me."),
    ),
    "habit:insist": (
        ("Understood. {task}. Starting tomorrow.", "Okay. If that's the directive, I'll do it."),
        ("A directive. Okay. I'm not happy about it.", "So it's not a conversation."),
    ),
    "habit:suggest": (
        ("I'll give it a shot.", "Okay. I'll try it in practice."),
        ("I'll think about it.", "Maybe."),
    ),
    "rolepact:path": (
        ("Okay. If there's a way back up, I can do {role} for now.", "That's fair. Show me the path and I'll take it."),
        ("Everyone says 'for now'.", "I'll believe the path when I see it."),
    ),
    "rolepact:honest": (
        ("I appreciate you being straight. I'll take {role}.", "Fair. At least I know the job."),
        ("That's a hard thing to hear.", "So that's what I am here."),
    ),
    "rolepact:value": (
        ("Okay. If the team needs it, I'll own it.", "That helps. I'll make {role} matter."),
        ("Selling it doesn't make it a bigger role.", "I'll do the job."),
    ),
}

HABIT_TASK = {
    "habit_shoot_more": "shooting more",
    "habit_shoot_less": "picking my shots",
    "habit_pass_more": "moving the puck quicker",
    "habit_pass_less": "taking those shots",
    "habit_play_physical": "finishing my checks",
    "habit_focus_defense": "thinking defence first",
    "habit_develop_playmaking": "working on my playmaking",
}
ROLE_NAME = {
    "rolepact_bottom_six": "the bottom six",
    "rolepact_third_pair": "the third pair",
    "rolepact_shutdown": "the shutdown job",
    "rolepact_pp_specialist": "the power-play job",
    "rolepact_backup_goalie": "the backup job",
}

# GM lines for meetings that had no written spoken line.
GM_SAY_EXTRA: Dict[str, Lines] = {
    "request_more_ice:honest": ("Here's the honest answer: you're at {toi} a night because the guys ahead of you are winning more battles. Win those and the minutes come.", "Honestly? You're close. Be harder to play against without the puck and you'll get more."),
    "request_more_ice:promise": ("You'll get a bigger role in the next five games. I'll talk to the coach today.", "I hear you. More minutes are coming in the next week or so. I'll make sure of it."),
    "request_more_ice:decline": ("The role gets earned on the ice, not in this office.", "Not right now. Keep playing like this and it'll force our hand."),
    "request_pp_time:honest": ("Right now there are four guys ahead of you on the power play. That's the honest picture.", "The units are set for now. Here's where you'd have to be to crack it."),
    "request_pp_time:promise": ("You'll get a real look on the power play within eight games. My word.", "Okay. I'll get you on the second unit soon. Make it count."),
    "request_pp_time:decline": ("The power play stays as it is.", "Not this time. I need you at five-on-five."),
    "request_starting_role:honest": ("Here's the plan in net: he's the starter, you get the back half of back-to-backs, and we look again in a month.", "Honestly, he's earned the net. Here's how you take it from him."),
    "request_starting_role:promise": ("You'll get a run of starts. Six games, and the net's yours to keep if you're good.", "Okay. Your run starts this week."),
    "request_starting_role:decline": ("We're staying with the starter.", "Not right now. He's the guy."),
    "contract_clarity:honest": ("Straight answer: you're in the plan, but I'm not talking numbers until the summer.", "Here's where we are: we want you back, and the number depends on the cap."),
    "contract_clarity:commit": ("You're part of the plan. That's not changing.", "I want you here long term. You can tell your agent that."),
    "contract_clarity:deflect": ("It's too early to talk about that.", "Let's revisit it later in the season."),
    "winning_concern_meeting:honest": ("Here's the plan: two more years of the young core growing, and we add when the window opens.", "Honestly, we're building. Here's what the next two years look like."),
    "winning_concern_meeting:promise_compete": ("I'm going to improve this roster before the deadline. You'll see moves.", "We're going for it. I'll be buying at the deadline."),
    "winning_concern_meeting:dismiss": ("Let me worry about the roster. You worry about your game.", "Focus on playing. The roster's my job."),
    "development_meeting:honest": ("Here's your plan: two things to work on, the coach who works with you on them, and a check-in in a month.", "We want you to become a top-six guy. Here's the path."),
    "development_meeting:skills_focus": ("I'm setting you up with the skills coach three mornings a week.", "We're going to work on one thing with the development staff. Start Monday."),
    "development_meeting:wait": ("Be patient. It's coming.", "Give it time. You're on the right track."),
    "habit:explain": ("Let me show you the video. Here's what happens when you're {task_ing}.", "Look at these numbers. When you're {task_ing}, you're a lot more dangerous."),
    "habit:insist": ("{task_gm}. That's a team directive, not a suggestion.", "I need you {task_ing}. Starting tomorrow."),
    "habit:suggest": ("Try {task_ing} in practice. See how it feels.", "Something to think about: {task_ing} might open your game up."),
    "rolepact_bottom_six:path": ("For now you're in the bottom six. Play it well and you'll work your way back up.",),
    "rolepact_bottom_six:honest": ("I'll be straight with you: right now you're a bottom-six forward for us.",),
    "rolepact_bottom_six:value": ("Our bottom six wins us games. I need someone I trust there.",),
    "rolepact_third_pair:path": ("For now you're on the third pair. Play it well and you'll move up.",),
    "rolepact_third_pair:honest": ("I'll be straight with you: right now you're a third-pair defenceman for us.",),
    "rolepact_third_pair:value": ("The third pair has to be steady or we lose games. I need you there.",),
    "rolepact_shutdown:path": ("I want you as our shutdown guy. Do it well and the offensive minutes come too.",),
    "rolepact_shutdown:honest": ("Your job is shutting down the other team's best line. That's it.",),
    "rolepact_shutdown:value": ("Shutting down the top line is the hardest job on the team. I want you doing it.",),
    "rolepact_pp_specialist:path": ("I want you running our power play. Do it well and the even-strength minutes follow.",),
    "rolepact_pp_specialist:honest": ("Your job is the power play. Even-strength minutes will be lighter.",),
    "rolepact_pp_specialist:value": ("Our power play has been dead. You're the guy who fixes it.",),
    "rolepact_backup_goalie:path": ("For now you're the backup. Play well when you go in and you'll get more starts.",),
    "rolepact_backup_goalie:honest": ("I'll be straight with you: you're our backup.",),
    "rolepact_backup_goalie:value": ("A good backup saves our season. I need you ready every time.",),
    "ask_mentor_younger:assign": ("There's a rookie who needs someone like you. I'm asking you to be that guy.",),
    "set_development_priority:shooting": ("Your shot. Get the release quicker and you'll score more.",),
    "set_development_priority:defence": ("Your own-zone game. That's what keeps you on the ice late.",),
    "set_development_priority:strength": ("The weight room. Win more battles and everything else follows.",),
    "set_development_priority:faceoffs": ("Faceoffs. Win the draw and you start with the puck.",),
    "set_development_priority:rebounds": ("Rebound control. Kill the second chances.",),
    "set_development_priority:positioning": ("Your positioning. Be square and let the puck hit you.",),
}

HABIT_GM = {
    "habit_shoot_more": ("Shoot more", "shooting more"),
    "habit_shoot_less": ("Pick your shots", "picking your shots"),
    "habit_pass_more": ("Move the puck quicker", "moving the puck quicker"),
    "habit_pass_less": ("Take those shots instead of passing", "taking those shots"),
    "habit_play_physical": ("Finish your checks", "finishing your checks"),
    "habit_focus_defense": ("Think defence first", "thinking defence first"),
    "habit_develop_playmaking": ("Work on your playmaking", "working on your playmaking"),
}

# GM openers for habit and role-pact talks: say the thing, so his reply makes sense.
HABIT_OPENERS = {
    "habit_shoot_more": ("I want you shooting more. You're passing up good looks.",),
    "habit_shoot_less": ("I want you picking your shots. Too many are going wide from bad spots.",),
    "habit_pass_more": ("I want you moving the puck quicker. You're holding onto it too long.",),
    "habit_pass_less": ("You're passing up shots you should be taking. I want you shooting those.",),
    "habit_play_physical": ("I want you playing more physical. Finish your checks.",),
    "habit_focus_defense": ("I want you thinking defence first for a while.",),
    "habit_develop_playmaking": ("I want you working on your playmaking this season.",),
    "rolepact_bottom_six": ("I need you to accept a bottom-six role for now.",),
    "rolepact_third_pair": ("I need you on the third pair for now.",),
    "rolepact_shutdown": ("I want you to be our shutdown guy.",),
    "rolepact_pp_specialist": ("I want you to be our power-play specialist.",),
    "rolepact_backup_goalie": ("I need you to accept the backup job.",),
}

# ---------------------------------------------------------------------------
# His specific ask, and the GM's last answer to it
# opts: (kind, button label, GM says, his goodbye)
# kind: yes / part / no
# ---------------------------------------------------------------------------

Opt = Tuple[str, str, str, str]

ASKS: Dict[str, Dict[str, object]] = {
    "discuss_ice_time": {"ask": ("So what does that mean for the next game? Same minutes?", "Give me a number. What should I expect a night?"),
        "opts": [("yes", "Promise more ice this week", "You'll see more ice this week. I'll talk to the coach.", "Alright. More ice this week. I'm holding you to it."),
                 ("part", "Tie it to the next three games", "Play the next three like you did tonight and the minutes go up.", "Three games. Okay. I'll make them count."),
                 ("no", "Same minutes for now", "For now, it's the same. Change my mind on the ice.", "Same minutes. Okay. I'll keep working.")]},
    "offer_increased_ice_time": {"ask": ("Where am I playing, though? Which line?", "Is it real minutes or just an extra shift here and there?"),
        "opts": [("yes", "Move him up a line", "You're moving up a line. Real minutes.", "Up a line. That's what I wanted. Thanks."),
                 ("part", "More shifts, same line", "More shifts where you are first. A line move comes after.", "More shifts first. Okay, I can work with that."),
                 ("no", "Don't get specific", "Let's see how the minutes go before we talk lines.", "Okay. We'll see what the sheet says.")]},
    "explain_reduced_ice_time": {"ask": ("So what gets them back? Give me something specific.", "How long is this going to last?"),
        "opts": [("yes", "Give him a clear target", "Win your battles on the wall and be reliable in your end. Do that for a week and the minutes come back.", "A week. Okay. I know what to do."),
                 ("part", "Say it's a few games", "Give it a few games. It's not permanent.", "A few games. Okay."),
                 ("no", "No timeline", "I'm not putting a timeline on it.", "No timeline. Got it.")]},
    "discuss_line_assignment": {"ask": ("Who am I playing with, then?", "So what's my job on that line? Score or shut it down?"),
        "opts": [("yes", "Give him the linemates he wants", "You'll be with guys who can finish. I'll set it up.", "Okay. Good linemates. That's all I needed."),
                 ("part", "Define the job, not the line", "Your job is to be hard to play against and chip in. The line will sort itself out.", "Hard to play against. Okay. I can do that."),
                 ("no", "Keep it open", "That's up to the coach. Be ready wherever he puts you.", "Okay. Wherever, then.")]},
    "discuss_promotion": {"ask": ("When? Give me a game.", "Is that a real promotion or a try-out?"),
        "opts": [("yes", "Next game", "Next game. You're moving up.", "Next game. I won't waste it."),
                 ("part", "After the road trip", "After this road trip. Keep it going.", "After the road trip. Okay. I'm counting."),
                 ("no", "No date", "When it's right. Don't watch the calendar.", "Okay. I'll keep pushing.")]},
    "discuss_demotion": {"ask": ("How long am I down there?", "What do I have to do to move back up?"),
        "opts": [("yes", "Two weeks and you're back", "Two weeks of good hockey and you're back up.", "Two weeks. Okay. Watch me."),
                 ("part", "When the details are back", "When the details are back, so are you.", "Okay. Details. Got it."),
                 ("no", "Until further notice", "Until I see something different. That's all I'll say.", "Okay. I'll make you change your mind.")]},
    "offer_pp_opportunity": {"ask": ("First unit or second?", "How many games do I get to show you?"),
        "opts": [("yes", "Second unit, real minutes", "Second unit, starting next game. Real minutes, not one shift.", "Second unit next game. I'll make it count."),
                 ("part", "Five games to show it", "You get five games. Produce and it's yours.", "Five games. Okay. That's fair."),
                 ("no", "Keep it vague", "We'll see how it goes. Be ready.", "Okay. I'll be ready.")]},
    "explain_pp_removal": {"ask": ("Can I at least get the second unit?", "When do I get another shot at it?"),
        "opts": [("yes", "Second unit for now", "Second unit for now. Earn your way back to the first.", "Second unit. Okay. I'll be back on the first."),
                 ("part", "A shot after the break", "After the break, we'll look at the units again.", "After the break. Okay."),
                 ("no", "No power play for now", "No power play for now. Win it back at five-on-five.", "Okay. Five-on-five, then.")]},
    "offer_pk_opportunity": {"ask": ("Is that on top of my regular shift or instead of it?", "First unit on the kill?"),
        "opts": [("yes", "On top of his minutes", "On top of your regular minutes. You've earned both.", "On top of. That's great. Thanks."),
                 ("part", "Second unit kill to start", "Second unit to start. Show the coaches and you move up.", "Second unit kill. Okay. I'll block everything."),
                 ("no", "Instead of some even-strength time", "It'll take some of your even-strength time. That's the trade-off.", "Okay. I'll take the kill.")]},
    "discuss_healthy_scratch": {"ask": ("When am I back in?", "What's the one thing you want to see?"),
        "opts": [("yes", "Back in next game", "You're back in next game.", "Next game. Okay. I'll be ready."),
                 ("part", "A couple of games", "A couple of games. Stay sharp in practice.", "A couple of games. Okay."),
                 ("no", "When the coach decides", "That's the coach's call. Stay ready.", "Okay. I'll stay ready.")]},
    "discuss_return_lineup": {"ask": ("Can I get some practice time on the line first?", "Is there a minutes limit for the first few games?"),
        "opts": [("yes", "Ease him in", "We'll ease you in. Lighter minutes for the first couple.", "Okay. Ease in. Makes sense."),
                 ("part", "Straight in, watch the minutes", "Straight in, but we'll watch your minutes.", "Okay. Straight in."),
                 ("no", "Straight in, full minutes", "Straight in, full minutes. We need you.", "Full minutes. Let's go.")]},
    "discuss_goalie_workload": {"ask": ("How many starts in a row before you look again?", "Do I get the big games?"),
        "opts": [("yes", "Six in a row", "Six starts in a row. Then we look again.", "Six. Okay. I'll make it hard to take back."),
                 ("part", "The next home stand", "The next home stand is yours.", "The home stand. Okay."),
                 ("no", "Game to game", "It's game to game. Be ready either way.", "Game to game. Okay.")]},
    "praise_performance": {"ask": ("Any chance that turns into more ice?", "Does that mean I get a look on the power play?"),
        "opts": [("yes", "More ice coming", "Yes. You've earned a bigger role.", "That's great. I won't let it slip."),
                 ("part", "Keep it up first", "Keep it up for a few more games and we'll talk.", "Okay. I'll keep it going."),
                 ("no", "Just enjoy it", "Just enjoy it. This was about recognition.", "Okay. Thanks.")]},
    "challenge_performance": {"ask": ("What do you want to see from me next game?", "Can I get some time with the skills coach?"),
        "opts": [("yes", "Give him a clear target", "Win more battles and get pucks to the net. Next game.", "Battles and pucks to the net. Got it."),
                 ("part", "Video with the coach", "Sit down with the coach and the video tomorrow.", "Okay. I'll be there."),
                 ("no", "You know what to do", "You know what to do.", "Yeah. I know what to do.")]},
    "ask_struggles": {"ask": ("Can I get a couple of games with guys I'm comfortable with?", "Can we keep this between us?"),
        "opts": [("yes", "Yes to both", "Yes. And this stays between us.", "Thanks. Really. That helps."),
                 ("part", "Between us, yes", "This stays between us. Lines are the coach's call.", "Okay. Thanks for keeping it quiet."),
                 ("no", "Push through it", "Push through it. That's how you get out of it.", "Okay. I'll push.")]},
    "discuss_scoring_drought": {"ask": ("Any chance I get some power-play time to get going?", "Can I keep my spot while I work through it?"),
        "opts": [("yes", "A power-play look", "You'll get some power-play time. Get one in and they'll come.", "Thanks. One goes in, the rest come."),
                 ("part", "You keep your spot", "You keep your spot. Keep shooting.", "Okay. Thanks."),
                 ("no", "No guarantees", "No guarantees. Produce and you keep it.", "Fair. I'll produce.")]},
    "discuss_defensive_struggles": {"ask": ("Can I get a partner who reads the game better?", "When's the video session?"),
        "opts": [("yes", "New partner and video", "New partner and video on Thursday.", "Okay. Thanks. I'll be better."),
                 ("part", "Video, same partner", "Video Thursday. Same partner for now.", "Okay. Thursday."),
                 ("no", "Just fix it", "Just clean it up.", "Alright. I'll clean it up.")]},
    "discuss_improvement": {"ask": ("Any chance that turns into a look on the power play?", "Can I keep playing with the same guys? It's working."),
        "opts": [("yes", "Give him a power-play look", "You've earned a look on the power play. Second unit, next game.", "Second unit. That's awesome. I'll keep it rolling."),
                 ("part", "Keep the line together", "The line stays together. The power play comes if you keep this up.", "Keeping the line works for me. I'll keep pushing."),
                 ("no", "Change nothing yet", "Let's not change anything. Just keep doing what you're doing.", "Fair enough. Don't fix what isn't broken.")]},
    "discuss_consistency": {"ask": ("Can someone help me put that routine together? I've never really had one.", "Can I get some feedback after games? Even just a text."),
        "opts": [("yes", "Set him up with the performance staff", "The performance staff will build it with you this week, and I'll check in after games.", "That would help a lot. Thanks."),
                 ("part", "Weekly check-in", "Let's sit down once a week and go over it.", "Once a week. Okay, I'll come prepared."),
                 ("no", "It's on him", "That's something you have to figure out yourself.", "Alright. I'll sort it out.")]},
    "discuss_playoff_performance": {"ask": ("Can I get some power-play time this series? I think I can help.", "Can you keep my line together? We've got something."),
        "opts": [("yes", "Power-play time this series", "You'll get power-play time this series.", "Let's go. I'm ready for this."),
                 ("part", "Keep the line together", "Your line stays together. The power play depends on matchups.", "Good. We'll do some damage."),
                 ("no", "No changes in the playoffs", "No changes now. Do your job and we'll be fine.", "Understood. I'll do my job.")]},
    "explain_ovr_rising": {"ask": ("Does that mean a bigger role at some point?", "Can I get more time with the skills coach to keep it going?"),
        "opts": [("yes", "More minutes now", "Yes. You've earned more minutes, starting now.", "That's huge. I won't slow down."),
                 ("part", "More skills time, role later", "More time with the skills coach, yes. The role comes with it.", "Fair. I'll keep stacking good weeks."),
                 ("no", "Just keep going", "Just keep doing what you're doing. Don't think about the role.", "Okay. Head down, then.")]},
    "explain_ovr_falling": {"ask": ("Can I get a couple of off days? My body's not right.", "Can I spend some extra time with the trainers?"),
        "opts": [("yes", "Rest and trainers", "Take two off days this week and spend them with the trainers.", "Thanks. My legs really need it."),
                 ("part", "Trainers, but play", "See the trainers, but I need you in the lineup.", "Okay. I'll play through it and see them after."),
                 ("no", "Play through it", "Everyone's banged up. Play through it.", "Alright. I'll grind it out.")]},
    "discuss_development_plan": {"ask": ("When do we check in on it? I want to know if I'm getting there.", "Who am I working with on this?"),
        "opts": [("yes", "Monthly with the skills coach", "Monthly check-ins with the skills coach and me.", "Monthly. Good. I'll have something to show you."),
                 ("part", "Review at season's end", "We'll review it properly at the end of the season.", "End of the season. Okay. I'll make it count."),
                 ("no", "Just put the work in", "Don't worry about check-ins. Just put the work in.", "Alright. I'll grind.")]},
    "set_development_priority": {"ask": ("Who do I work with on it?", "How do we know if it's working?"),
        "opts": [("yes", "Skills coach, monthly check", "You'll work with the skills coach, and we'll check the numbers every month.", "Monthly check-ins. Perfect. I'll be on it."),
                 ("part", "Check at the break", "We'll look at it again at the All-Star break.", "The break. Okay. Give me till then."),
                 ("no", "Do it on your own time", "Put the work in on your own time.", "Alright. Early mornings, then.")]},
    "discuss_stalled_development": {"ask": ("What kind of opportunity are we talking about?", "Can I get more minutes to break through? I can't grow on the bench."),
        "opts": [("yes", "More minutes next week", "More minutes, starting next week.", "That's all I need. Thanks."),
                 ("part", "A special-teams look", "A look on special teams first.", "Special teams. I'll take it and run."),
                 ("no", "Keep being patient", "Be patient. The opportunity comes when it comes.", "Okay. I'll try to stay patient.")]},
    "discuss_aging_decline": {"ask": ("Will I still be out there in the big moments?", "Do I still dress every night?"),
        "opts": [("yes", "Still out there when it matters", "When the game's on the line, you're still out there.", "That's all I wanted. Thanks for that."),
                 ("part", "Fewer back-to-backs", "You'll sit some back-to-backs, but you're still a key guy.", "Fewer back-to-backs. I can live with that."),
                 ("no", "No promises", "I can't promise that. It'll depend on how you're playing.", "Understood. I'll make you want me out there.")]},
    "discuss_conditioning": {"ask": ("Can I skip a morning skate on back-to-backs?", "Can I see the trainers this week? Something's off."),
        "opts": [("yes", "Yes to both", "Skip morning skates on back-to-backs, and see the trainers tomorrow.", "Thanks. I'll be fresher, I promise."),
                 ("part", "Trainers, but skate", "See the trainers. Morning skates stay.", "Fair. I'll see them tomorrow."),
                 ("no", "Same as everyone", "No. Everyone does the same thing.", "Alright. Same as everyone.")]},
    "discuss_injury_recovery": {"ask": ("Can I travel with the team while I rehab? I hate being away from the guys.", "Can I start skating with the group?"),
        "opts": [("yes", "Travel with the team", "Travel with the team. The trainers will come too.", "Thanks. Being around the guys helps."),
                 ("part", "Skate with the group, stay home", "Skate with the group, but stay home on road trips.", "Okay. At least I'll be on the ice with them."),
                 ("no", "Stick to the rehab plan", "Stick to the rehab plan. No shortcuts.", "Alright. I'll do it right.")]},
    "ask_extension_interest": {"ask": ("Are you thinking term or money first?", "Can we get it done before the deadline? I don't want it hanging over me."),
        "opts": [("yes", "Done before the deadline", "Let's get it done before the deadline.", "Before the deadline. Good. I'll tell my agent."),
                 ("part", "Aim for the summer", "Let's aim for the summer.", "Summer. Okay. As long as we're talking."),
                 ("no", "No timeline yet", "I'm not putting a timeline on it.", "No timeline. I'll let my agent know.")]},
    "long_term_plans": {"ask": ("Does that mean an extension's coming?", "Can I tell my agent you said that?"),
        "opts": [("yes", "Tell your agent", "Tell your agent. He'll hear from us.", "Good. I'll let him know tonight."),
                 ("part", "When the time's right", "When the time's right. You'll be the first to know.", "Fair. I'll trust you on that."),
                 ("no", "Keep it between us", "Keep this between us for now.", "Okay. It stays in this room.")]},
    "future_uncertain": {"ask": ("Will you tell me before anything happens?", "Should I tell my agent to talk to other teams?"),
        "opts": [("yes", "You'll hear it from me first", "You'll hear it from me first. Always.", "Thanks. That matters to me."),
                 ("part", "Talk to your agent", "Talk to your agent. That's fair.", "Okay. I'll have him look around."),
                 ("no", "Focus on playing", "Focus on playing. The rest takes care of itself.", "Alright. I'll try.")]},
    "discuss_role_next_season": {"ask": ("Can you put that in writing?", "Do I keep my power-play time?"),
        "opts": [("yes", "It's a commitment", "Consider it a commitment. You'll have it.", "Good. I'll plan my summer around it."),
                 ("part", "It gets settled in camp", "Training camp will settle it.", "Camp. Okay. I'll show up ready."),
                 ("no", "No promises", "I can't promise anything for next year.", "Okay. Then I'll keep my options open.")]},
    "discuss_upcoming_fa": {"ask": ("When can my agent expect an offer?", "Can we get something done before the deadline?"),
        "opts": [("yes", "Offer this month", "You'll have an offer this month. I'll call your agent.", "This month. That's good to hear."),
                 ("part", "After the deadline", "After the deadline. I need to see where we are.", "After the deadline. Okay."),
                 ("no", "We'll see", "We'll see how the season plays out.", "Then I'll see what July looks like.")]},
    "hometown_discount": {"ask": ("What do I get for it? More term? A no-trade clause?", "How much less are we talking?"),
        "opts": [("yes", "Term and a no-trade clause", "Extra term and a no-trade clause.", "That I'll think about seriously. Thanks."),
                 ("part", "An extra year", "An extra year of term.", "An extra year. Let me talk to my agent."),
                 ("no", "Nothing extra", "Nothing extra. It's about staying here.", "Then I need to think about it.")]},
    "contract_expectations": {"ask": ("So where are you on term?", "What's your number, roughly?"),
        "opts": [("yes", "Term, yes", "We can do term.", "Good. That's what I wanted."),
                 ("part", "Shorter term, higher number", "Shorter term, a little more per year.", "Let me think about that one."),
                 ("no", "Numbers later", "We'll talk numbers later.", "Okay. My agent will be waiting.")]},
    "discuss_retirement": {"ask": ("Is there still a role for me here if I keep going?", "Can I take the summer to decide?"),
        "opts": [("yes", "A role and the summer", "There's a role for you, and take the summer.", "Thanks. That makes it easier."),
                 ("part", "Take the summer", "Take the summer. We'll talk in July.", "July. Okay. I'll think it over."),
                 ("no", "We need an answer soon", "I need to know soon, for planning.", "Okay. I'll give you an answer.")]},
    "ask_trade_welcome": {"ask": ("Can I give you a list of teams?", "Will you keep this quiet?"),
        "opts": [("yes", "Take the list, keep it quiet", "Send me a list. It stays quiet.", "Thanks. I'll have it to you this week."),
                 ("part", "Quiet, no list", "It stays quiet. I can't promise teams.", "Okay. As long as it's quiet."),
                 ("no", "No promises", "I can't promise anything.", "Then I'll be ready for anything.")]},
    "tell_being_shopped": {"ask": ("Will you tell me before it happens?", "Do I get a say in where I go?"),
        "opts": [("yes", "A heads-up and a say", "You'll get a heads-up, and I'll listen on where.", "Thanks. That's fair."),
                 ("part", "A heads-up", "You'll get a heads-up before anything's done.", "Okay. That helps a little."),
                 ("no", "Business is business", "I can't promise that. That's the business.", "Then I'll start preparing my family.")]},
    "reassure_not_shopped": {"ask": ("Can I tell my family that?", "And if that changes?"),
        "opts": [("yes", "Tell them", "Tell them. You're staying.", "Thanks. They'll be relieved."),
                 ("part", "You'll hear it from me", "If anything changes, you'll hear it from me.", "Okay. I'll hold you to that."),
                 ("no", "Don't overthink it", "Don't overthink it.", "Easier said than done, but okay.")]},
    "ask_ntc_waiver": {"ask": ("Which teams are we talking about?", "What do I get if I say yes?"),
        "opts": [("yes", "Contenders only", "Contenders only, from a list you pick.", "Okay. I'll look at the list."),
                 ("part", "A short list", "I'll send you a short list.", "Send it. I'm not promising anything."),
                 ("no", "Just think about it", "Just think about it.", "I'll think about it. Don't count on it.")]},
    "ask_preferred_destinations": {"ask": ("Will you actually use the list?", "Will you keep it between us?"),
        "opts": [("yes", "Yes, and it's private", "Yes, and it stays in this room.", "Thanks. I'll send it tomorrow."),
                 ("part", "I'll try", "I'll do my best with it.", "Okay. That's fair."),
                 ("no", "No promises", "I can't promise I'll use it.", "Then why ask?")]},
    "discuss_ahl_assignment": {"ask": ("How long am I down there?", "Will I play big minutes and the power play?"),
        "opts": [("yes", "Two weeks, big minutes", "Two weeks. Big minutes, both special teams.", "Two weeks. I'll be back."),
                 ("part", "Until you're ready", "Until you're ready. The coaches will tell me.", "Okay. I'll make it quick."),
                 ("no", "No timeline", "No timeline.", "No timeline. Okay.")]},
    "discuss_call_up": {"ask": ("When? Give me something to aim at.", "Will I actually play when I get up there?"),
        "opts": [("yes", "Next injury, you're first", "Next injury up front, you're the first call.", "First call. I'll be ready."),
                 ("part", "After the break", "After the break, we'll look.", "After the break. Okay."),
                 ("no", "No date", "I'm not giving you a date.", "Okay. I'll keep pushing.")]},
    "address_trade_rumors": {"ask": ("Will you tell me if that changes?", "Can I say that to the media when they ask?"),
        "opts": [("yes", "Yes to both", "Yes. And you can tell them we talked.", "Thanks. That shuts it down."),
                 ("part", "You'll hear it from me", "If anything changes, you'll hear it from me.", "Okay. I'll hold you to that."),
                 ("no", "Say nothing to the media", "Say nothing to the media.", "Okay. No comment it is.")]},
    "ask_coaching_relationship": {"ask": ("Can you set it up this week?", "Will you be in the room?"),
        "opts": [("yes", "This week, I'll be there", "This week, and I'll be in the room.", "Thanks. That'll help."),
                 ("part", "Next week", "Next week. Let things cool off first.", "Next week. Okay."),
                 ("no", "Work it out yourselves", "Work it out between the two of you.", "Okay. I'll talk to him.")]},
    "ask_locker_room_concerns": {"ask": ("Can I run a players-only meeting?", "Will you take it to the leadership group?"),
        "opts": [("yes", "Run the meeting", "Run it after practice tomorrow.", "Okay. I'll set it up."),
                 ("part", "I'll talk to the leaders", "I'll talk to the leadership group this week.", "Thanks. Leave my name out of it."),
                 ("no", "Not right now", "Not right now. Let it play out.", "Okay. Let me know if it gets worse.")]},
    "discuss_leadership": {"ask": ("Is there a letter in that?", "Can I sit in on the leadership meetings?"),
        "opts": [("yes", "An A this season", "You'll wear an A this season.", "That means a lot. Thank you."),
                 ("part", "Leadership meetings", "You're in the leadership meetings from now on.", "Thanks. I'll make sure my voice is heard."),
                 ("no", "Lead without it", "You don't need a letter to lead.", "Fair. I'll lead anyway.")]},
    "ask_mentor_younger": {"ask": ("Who is it?", "Can you move his stall next to mine?"),
        "opts": [("yes", "Name the rookie, move his stall", "The rookie. I'll move his stall next to yours.", "Okay. I'll take care of him."),
                 ("part", "Name the rookie", "The rookie. Keep an eye on him.", "Okay. I'll check in on him."),
                 ("no", "Whoever needs it", "Whoever needs it. You'll know.", "Alright. I'll look out for the young guys.")]},
    "personal_wellbeing_check": {"ask": ("Can I take a day if I need one?", "Can we keep this private?"),
        "opts": [("yes", "Yes to both", "Yes, and it stays between us.", "Thanks. Really."),
                 ("part", "It stays private", "It stays between us.", "Thank you. That means a lot."),
                 ("no", "Let's see", "Let's see how it goes.", "Okay. I'll let you know.")]},
    "repair_relationship": {"ask": ("Can we talk more often? I'd rather not go through my agent.", "Can I come to you directly when something's wrong?"),
        "opts": [("yes", "Door's always open", "Door's always open. Come straight to me.", "Okay. Thanks. I will."),
                 ("part", "Monthly check-ins", "Let's check in once a month.", "Once a month. That's a start."),
                 ("no", "Go through the coach", "Go through the coach first.", "Okay. I'll try that.")]},
    "address_social_post": {"ask": ("Can I at least explain it myself?", "Is this going to come up with the media?"),
        "opts": [("yes", "Explain it yourself", "Explain it yourself after practice tomorrow.", "Okay. I'll handle it the right way."),
                 ("part", "I'll handle the media", "I'll handle the media. You stay quiet.", "Thanks. I'll stay off it."),
                 ("no", "Say nothing", "Say nothing. Let it die.", "Okay. Radio silence.")]},
    "request_more_ice": {"ask": ("So what's the timeline?", "How many games do I get to show you?"),
        "opts": [("yes", "Next week", "Next week. You'll see it on the sheet.", "Next week. I'll be ready."),
                 ("part", "After a few more games", "After a few more games like this.", "A few more. Okay, I'll keep it up."),
                 ("no", "No timeline", "I'm not putting a timeline on it.", "Okay. I'll keep earning it.")]},
    "request_pp_time": {"ask": ("Second unit to start?", "How many games do I get?"),
        "opts": [("yes", "Second unit", "Second unit, starting next game.", "Next game. I'll make it count."),
                 ("part", "Five games to show it", "Five games. Produce and you stay.", "Five games. That's fair."),
                 ("no", "Not this season", "Not this season.", "Okay. I'll keep showing you.")]},
    "request_starting_role": {"ask": ("How many games?", "Can I get the next home stand?"),
        "opts": [("yes", "The home stand", "The next home stand is yours.", "The home stand. Thanks. I'll take it."),
                 ("part", "Three starts", "Three starts. Then we'll see.", "Three. Okay, I'll make them count."),
                 ("no", "No change", "No change for now.", "Okay. I'll be ready when you need me.")]},
    "contract_clarity": {"ask": ("When can my agent expect a call?", "Can I tell him I'm in the plan?"),
        "opts": [("yes", "A call this month", "Your agent gets a call this month.", "Good. I'll let him know."),
                 ("part", "Tell him you're in the plan", "Tell him you're in the plan. Numbers later.", "Okay. That's something."),
                 ("no", "Not yet", "Nothing to tell him yet.", "Okay. I'll keep waiting.")]},
    "winning_concern_meeting": {"ask": ("Are we buyers at the deadline?", "When do things actually change?"),
        "opts": [("yes", "We'll be buyers", "We'll be buyers at the deadline.", "Good. That's all I wanted to hear."),
                 ("part", "Depends on the standings", "Depends where we are in February.", "Okay. Let's make sure we're in it."),
                 ("no", "Not this year", "Not this year. We're building.", "That's hard to hear. Okay.")]},
    "development_meeting": {"ask": ("When do we check in?", "Who am I working with?"),
        "opts": [("yes", "Monthly, skills coach", "Monthly, with the skills coach and me.", "Good. I'll have something to show you."),
                 ("part", "At the break", "We'll check in at the break.", "Okay. I'll be ready."),
                 ("no", "Just work hard", "Just work hard. It'll come.", "Okay. I'll grind.")]},
}

# Habit / role-pact talks.
HABIT_ASK: Dict[str, object] = {"ask": ("How long do you want me doing it?", "How long are we trying this?"),
    "opts": [("yes", "Try it for a month", "Try it for a month. If it doesn't work, we drop it.", "A month. Okay. I'll give it a real try."),
             ("part", "Until it clicks", "Until it clicks.", "Until it clicks. I'll give it a real shot."),
             ("no", "Permanently", "This is how we play now.", "Okay. If that's how we play, I'll adjust.")]}
ROLEPACT_ASK: Dict[str, object] = {"ask": ("For how long?", "Is there a way back up?"),
    "opts": [("yes", "Until the break, then we review", "Until the break. Then we look at it again.", "Until the break. Okay. I can do that."),
             ("part", "If you play well, you move up", "If you play well, you'll move up.", "Okay. I'll play my way back up."),
             ("no", "Indefinitely", "This is the job.", "Then I'll do the job as well as I can.")]}

# Tempers flared: closing lines keyed by the GM's last move, then by how it landed.
HOT_CLOSE: Dict[str, Dict[str, Lines]] = {
    "cool": {"good": ("...Okay. I'm calm. But I meant what I said.", "Alright. Sorry. It's been building."),
             "bad": ("Fine. I'm sitting. Doesn't change anything.", "Okay. I'm calm. I'm still not happy.")},
    "ground": {"good": ("...Fine. You're right. I'll watch my tone.", "Okay. Point taken."),
               "bad": ("You don't get to talk to me like that.", "We're done here.")},
    "flatter": {"good": ("...Yeah. Okay. Thanks for saying that.", "That helps. A little."),
                "bad": ("Don't butter me up.", "I've heard that before.")},
    "bluff": {"good": ("...No. I'm staying. Let's figure this out.", "Okay. I'll sit back down."),
              "bad": ("Fine. I'm going.", "Then I'm going.")},
}
