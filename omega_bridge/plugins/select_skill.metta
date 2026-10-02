(= (select-skill $skill_id)
   (py-call (subrep_channel.select_skill $skill_id)))

(= (loadOmegaPlugin)
   (add-skill select-skill
     "When you receive a skill selection request, respond with EXACTLY: select-skill <skill_id> — one word, no quotes, nothing else. Example: select-skill skill_001_ppo_deterministic"
     (skill_id)))