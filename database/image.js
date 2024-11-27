const fs = require('fs');
const mysql = require('mysql');

// Create a connection to the MySQL server
const connection = mysql.createConnection({
  host: 'localhost',
  user: 'root', // Change to your MySQL username
  password: '', // Change to your MySQL password
  database: 'semenggoh' // Specify the database name here
});

// Connect to the MySQL server
connection.connect((err) => {
  if (err) throw err;
  console.log('Connected to MySQL server.');

  // Select image data from the images table
  const selectImageQuery = `SELECT image_name, image_data FROM images WHERE image_id = ?`;
  const imageId = 1; // Specify the image_id of the image you want to retrieve

  connection.query(selectImageQuery, [imageId], (err, result) => {
    if (err) throw err;

    if (result.length > 0) {
      const { image_name, image_data } = result[0];
      const filePath = `./retrieved_${image_name}`;

      // Write the binary data to a file
      fs.writeFileSync(filePath, image_data);
      console.log(`Image saved as ${filePath}`);
    } else {
      console.log('No image found with the specified ID.');
    }

    // Close the connection
    connection.end((err) => {
      if (err) throw err;
      console.log('Connection closed.');
    });
  });
});
