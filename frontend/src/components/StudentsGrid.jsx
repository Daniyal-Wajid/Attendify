
import { students } from "../data/mockStudents";
import StudentCard from "./StudentCard";

export default function StudentsGrid() {
  return (
    <div className="grid grid-cols-4 gap-6">
      {students.map((student) => (
        <StudentCard key={student.id} student={student} />
      ))}
    </div>
  );
}
