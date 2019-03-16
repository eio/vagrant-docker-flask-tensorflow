from app import db, students

db.create_all()

test_rec = students(
    'Donald Duck',
    'New Angeles',
    '1 Infinite Mirror'
)

db.session.add(test_rec)
db.session.commit()
