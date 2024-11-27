const mysql = require('mysql');

// Create a connection to the MySQL server
const connection = mysql.createConnection({
  host: 'localhost',
  user: 'root', // Change to your MySQL username
  password: '', // Change to your MySQL password
});

// Connect to the MySQL server
connection.connect((err) => {
  if (err) throw err;
  console.log('Connected to MySQL server.');

  // Create the database if it doesn't exist
  connection.query('CREATE DATABASE IF NOT EXISTS semenggoh', (err, result) => {
    if (err) throw err;
    console.log('Database created or already exists.');

    // Use the semenggoh database
    connection.query('USE semenggoh', (err) => {
      if (err) throw err;
      console.log('Using the semenggoh database.');

      // Create the sites table
      const createSitesTable = `
        CREATE TABLE IF NOT EXISTS sites (
          site_id INT AUTO_INCREMENT PRIMARY KEY,
          site_name VARCHAR(255) NOT NULL,
          latitude FLOAT NOT NULL,
          longitude FLOAT NOT NULL,
          division VARCHAR(255) NOT NULL
        )`;
      connection.query(createSitesTable, (err) => {
        if (err) throw err;
        console.log('Sites table created or already exists.');
      });

      // Create the updated images table with separate date and time columns
      const createImagesTable = `
        CREATE TABLE IF NOT EXISTS images (
          image_id INT AUTO_INCREMENT PRIMARY KEY,
          image_name VARCHAR(255) NOT NULL,
          image_data LONGBLOB NOT NULL,
          capture_date DATE NOT NULL,
          capture_time TIME NOT NULL,
          temperature FLOAT NOT NULL,
          site_id INT,
          FOREIGN KEY (site_id) REFERENCES sites(site_id)
        )`;
      connection.query(createImagesTable, (err) => {
        if (err) throw err;
        console.log('Images table created or already exists.');
      });

      // Create the species table
      const createSpeciesTable = `
        CREATE TABLE IF NOT EXISTS species (
          species_id INT PRIMARY KEY,
          species_name VARCHAR(255) NOT NULL
        )`;
      connection.query(createSpeciesTable, (err) => {
        if (err) throw err;
        console.log('Species table created or already exists.');
      });

      // Create the predictions table
      const createPredictionsTable = `
        CREATE TABLE IF NOT EXISTS predictions (
          prediction_id INT AUTO_INCREMENT PRIMARY KEY,
          image_id INT,
          species_id INT,
          confidence_score FLOAT NOT NULL,
          FOREIGN KEY (image_id) REFERENCES images(image_id),
          FOREIGN KEY (species_id) REFERENCES species(species_id)
        )`;
      connection.query(createPredictionsTable, (err) => {
        if (err) throw err;
        console.log('Predictions table created or already exists.');
      });

      // Close the connection
      connection.end((err) => {
        if (err) throw err;
        console.log('Connection closed.');
      });
    });
  });
});
