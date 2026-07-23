/** True when the user's franchise is in the playoff field (or we cannot tell yet). */
export function userMadePlayoffs(franchiseState) {
  if (!franchiseState) return true;

  const uid = String(
    franchiseState.user_team_id ||
      franchiseState.userTeamId ||
      franchiseState.team?.id ||
      franchiseState.team?.team_id ||
      ""
  )
    .trim()
    .toLowerCase();
  if (!uid) return true;

  const payload =
    franchiseState.playoff_payload ||
    franchiseState.playoff_data ||
    franchiseState.playoffs ||
    {};
  const teamLists = [
    payload.teams,
    payload.playoff_teams,
    payload.playoffTeams,
    franchiseState.playoffs?.teams,
    franchiseState.playoffs?.playoff_teams,
  ].filter(Array.isArray);

  for (const list of teamLists) {
    if (
      list.some(
        (team) =>
          String(team?.team_id || team?.teamId || team?.id || "")
            .trim()
            .toLowerCase() === uid
      )
    ) {
      return true;
    }
  }

  const standings = franchiseState.standings;
  if (Array.isArray(standings)) {
    const row = standings.find(
      (team) =>
        String(team?.team_id || team?.teamId || team?.id || "")
          .trim()
          .toLowerCase() === uid
    );
    if (row) {
      if (row.is_playoff_team === true) return true;
      const status = String(row.playoff_status || "").toLowerCase();
      if (status === "clinched" || status === "playoff") return true;
      if (row.is_eliminated === true || status === "eliminated") return false;
    }
  }

  if (teamLists.some((list) => list.length > 0)) return false;

  return true;
}
