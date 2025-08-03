const express = require('express');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = 3000;
const USERS_FILE = path.join(__dirname, 'users.json');

// Middleware
app.use(bodyParser.json());
app.use(express.static(__dirname)); // <-- serve HTML files from current directory

const readUsers = () => JSON.parse(fs.readFileSync(USERS_FILE, 'utf-8'));
const writeUsers = (users) => fs.writeFileSync(USERS_FILE, JSON.stringify(users, null, 2));

// Signup route
app.post('/signup', (req, res) => {
  const { name, email, password } = req.body;
  const users = readUsers();

  if (users.find(user => user.email === email)) {
    return res.status(409).json({ message: 'User already exists' });
  }

  users.push({ name, email, password });
  writeUsers(users);
  res.status(201).json({ message: 'Signup successful' });
});

// Login route
app.post('/login', (req, res) => {
  const { email, password } = req.body;
  const users = readUsers();

  const user = users.find(u => u.email === email && u.password === password);
  if (user) {
    res.json({ message: 'Login successful', name: user.name });
  } else {
    res.status(401).json({ message: 'Invalid credentials' });
  }
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
