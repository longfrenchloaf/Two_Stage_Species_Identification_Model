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

  // Insert example data into the sites table
  const insertSites = `
    INSERT INTO sites (site_name, latitude, longitude, division) VALUES
    ('Site A', 1.4854, 110.3548, 'Kuching'),
    ('Site B', 1.4507, 110.4573, 'Bau'),
    ('Site C', 1.3962, 110.4692, 'Serian')`;
  connection.query(insertSites, (err) => {
    if (err) throw err;
    console.log('Example data inserted into sites table.');
  });

  // Function to read the image file as binary data
  const readImageAsBinary = (filePath) => {
    return fs.readFileSync(filePath);
  };

  // Insert example data into the images table
  const insertImages = `
    INSERT INTO images (image_name, image_data, capture_date_time, temperature, site_id, other_metadata) VALUES (?, ?, ?, ?, ?, ?)`;

  // Array of images with metadata
  const images = [
    {
      filePath: 'C:\\Users\\Brenda\\Documents\\COS30049 Computing Technology Innovation Project\\yolov8_resnet\\database\\ANIMAL1.jpg',
      imageName: 'site_a_01.jpg',
      captureDateTime: '2024-10-01 12:30:00',
      temperature: 28.5,
      siteId: 1,
      otherMetadata: '{"weather": "sunny"}'
    },
    {
      filePath: 'C:\\Users\\Brenda\\Documents\\COS30049 Computing Technology Innovation Project\\yolov8_resnet\\database\\ANIMAL2.jpg',
      imageName: 'site_b_01.jpg',
      captureDateTime: '2024-10-02 14:45:00',
      temperature: 30.2,
      siteId: 2,
      otherMetadata: '{"weather": "cloudy"}'
    },
    {
      filePath: 'C:\\Users\\Brenda\\Documents\\COS30049 Computing Technology Innovation Project\\yolov8_resnet\\database\\ANIMAL3.jpg',
      imageName: 'site_c_01.jpg',
      captureDateTime: '2024-10-03 16:00:00',
      temperature: 27.8,
      siteId: 3,
      otherMetadata: '{"weather": "rainy"}'
    }
  ];

  // Insert each image into the database
  images.forEach((img) => {
    const imageData = readImageAsBinary(img.filePath);
    connection.query(insertImages, [img.imageName, imageData, img.captureDateTime, img.temperature, img.siteId, img.otherMetadata], (err) => {
      if (err) throw err;
      console.log(`Image data for ${img.imageName} inserted into images table.`);
    });
  });

  // Insert example data into the species table
  const insertSpecies = `
    INSERT INTO species (species_name) VALUES
    ('Orangutan'),
    ('Proboscis Monkey'),
    ('Bornean Gibbon')`;
  connection.query(insertSpecies, (err) => {
    if (err) throw err;
    console.log('Example data inserted into species table.');
  });

  // Insert example data into the predictions table
  const insertPredictions = `
    INSERT INTO predictions (image_id, species_id, confidence_score) VALUES
    (1, 1, 0.95),
    (2, 2, 0.85),
    (3, 3, 0.90)`;
  connection.query(insertPredictions, (err) => {
    if (err) throw err;
    console.log('Example data inserted into predictions table.');
  });

  // Close the connection
  connection.end((err) => {
    if (err) throw err;
    console.log('Connection closed.');
  });
});
