import { policyRules } from "../data/mockPolicyRules";
import PolicyRuleCard from "./PolicyRuleCard";

export default function PolicyRulesGrid() {
  return (
    <div className="grid grid-cols-3 gap-6">
      {policyRules.map((rule) => (
        <PolicyRuleCard key={rule.id} rule={rule} />
      ))}
    </div>
  );
}
